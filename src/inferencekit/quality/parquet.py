"""A small, dependency-free Parquet reader: just enough to read a flat table of string columns.

It exists so ``quality gsm8k`` can read the Hugging Face copy of GSM8K (a Parquet file)
without pulling in pyarrow. Supported: flat schemas (no nested or repeated columns),
BYTE_ARRAY / INT32 / INT64 columns, REQUIRED or OPTIONAL, PLAIN and dictionary encodings
(PLAIN_DICTIONARY / RLE_DICTIONARY), data pages v1 and v2, and the UNCOMPRESSED, SNAPPY
and GZIP codecs. Anything else raises ``ParquetError`` instead of returning wrong data.

Format reference: github.com/apache/parquet-format (parquet.thrift, Encodings.md); the
footer and page headers use the Thrift compact protocol; Snappy is the raw block format.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


class ParquetError(ValueError):
    pass


# ----------------------------------------------------------------------------- thrift compact protocol

class _Thrift:
    """Reads Thrift compact-protocol structs as {field_id: value} dicts."""

    def __init__(self, buf: bytes, pos: int = 0):
        self.buf, self.pos = buf, pos

    def byte(self) -> int:
        if self.pos >= len(self.buf):
            raise ParquetError("truncated thrift data")
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def varint(self) -> int:
        shift = result = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def binary(self) -> bytes:
        n = self.varint()
        out = self.buf[self.pos:self.pos + n]
        if len(out) != n:
            raise ParquetError("truncated thrift binary")
        self.pos += n
        return out

    def value(self, t: int):
        if t in (1, 2):  # bool inside a container: one byte, 1 = true
            return self.byte() == 1
        if t == 3:
            return struct.unpack("b", bytes([self.byte()]))[0]
        if t in (4, 5, 6):
            return self.zigzag()
        if t == 7:
            out = struct.unpack_from("<d", self.buf, self.pos)[0]
            self.pos += 8
            return out
        if t == 8:
            return self.binary()
        if t in (9, 10):
            head = self.byte()
            size, et = head >> 4, head & 0x0F
            if size == 15:
                size = self.varint()
            return [self.value(et) for _ in range(size)]
        if t == 11:
            size = self.varint()
            if not size:
                return {}
            kv = self.byte()
            return {self.value(kv >> 4): self.value(kv & 0x0F) for _ in range(size)}
        if t == 12:
            return self.struct()
        raise ParquetError(f"unknown thrift type {t}")

    def struct(self) -> dict:
        out, last = {}, 0
        while True:
            head = self.byte()
            if head == 0:
                return out
            delta, t = head >> 4, head & 0x0F
            fid = last + delta if delta else self.zigzag()
            last = fid
            out[fid] = True if t == 1 else False if t == 2 else self.value(t)


# ----------------------------------------------------------------------------- codecs and encodings

def snappy_decompress(data: bytes) -> bytes:
    """Raw (unframed) Snappy block decompression."""
    t = _Thrift(data)
    size = t.varint()
    out = bytearray()
    pos, n = t.pos, len(data)
    while pos < n:
        tag = data[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:  # literal
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(data[pos:pos + nb], "little")
                pos += nb
            ln += 1
            out += data[pos:pos + ln]
            pos += ln
            continue
        if kind == 1:
            ln = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | data[pos]
            pos += 1
        elif kind == 2:
            ln = (tag >> 2) + 1
            off = int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
        if off <= 0 or off > len(out):
            raise ParquetError("corrupt snappy data (bad copy offset)")
        start = len(out) - off
        if off >= ln:
            out += out[start:start + ln]
        else:  # overlapping copy: repeat the last `off` bytes
            for i in range(ln):
                out.append(out[start + i])
    if len(out) != size:
        raise ParquetError(f"corrupt snappy data ({len(out)} bytes, header says {size})")
    return bytes(out)


def _decompress(codec: int, data: bytes, size: int) -> bytes:
    if codec == 0:
        return data
    if codec == 1:
        return snappy_decompress(data)
    if codec == 2:
        return zlib.decompress(data, 47)  # gzip or zlib header
    raise ParquetError(f"unsupported compression codec {codec} (supported: none, snappy, gzip)")


def _hybrid(buf: bytes, pos: int, end: int, width: int, count: int) -> list[int]:
    """RLE / bit-packed hybrid decoding of ``count`` values of ``width`` bits from buf[pos:end]."""
    out: list[int] = []
    if width == 0:
        return [0] * count
    nbytes = (width + 7) // 8
    mask = (1 << width) - 1
    t = _Thrift(buf, pos)
    while len(out) < count and t.pos < end:
        head = t.varint()
        if head & 1:  # bit-packed: (head >> 1) groups of 8 values
            ngroups = head >> 1
            raw = int.from_bytes(buf[t.pos:t.pos + ngroups * width], "little")
            t.pos += ngroups * width
            out.extend((raw >> (i * width)) & mask for i in range(ngroups * 8))
        else:  # run of one value
            value = int.from_bytes(buf[t.pos:t.pos + nbytes], "little")
            t.pos += nbytes
            out.extend([value] * (head >> 1))
    if len(out) < count:
        raise ParquetError("RLE data ended early")
    return out[:count]


def _plain(ptype: int, buf: bytes, pos: int, count: int) -> list:
    if ptype == 6:  # BYTE_ARRAY
        out = []
        for _ in range(count):
            (ln,) = struct.unpack_from("<I", buf, pos)
            out.append(bytes(buf[pos + 4:pos + 4 + ln]))
            pos += 4 + ln
        return out
    if ptype in (1, 2):  # INT32, INT64
        fmt = "<i" if ptype == 1 else "<q"
        return list(struct.unpack_from(f"<{count}{fmt[1]}", buf, pos))
    raise ParquetError(f"unsupported physical type {ptype}")


# ----------------------------------------------------------------------------- reading

def _column(data: bytes, meta: dict, ptype: int, max_def: int) -> list:
    codec, total = meta.get(4, 0), meta.get(5, 0)
    first_data, first_dict = meta.get(9), meta.get(11)
    start = first_dict if first_dict and (first_data is None or first_dict < first_data) else first_data
    dictionary = None
    values: list = []
    pos = start
    while len(values) < total:
        t = _Thrift(data, pos)
        header = t.struct()
        pos = t.pos
        ptype_page, usize, csize = header.get(1), header.get(2, 0), header.get(3, 0)
        body = data[pos:pos + csize]
        pos += csize
        if ptype_page == 2:  # dictionary page
            dh = header.get(7) or {}
            dictionary = _plain(ptype, _decompress(codec, body, usize), 0, dh.get(1, 0))
            continue
        if ptype_page == 0:  # data page v1: [def levels][values], all compressed together
            dh = header.get(5) or {}
            n, enc = dh.get(1, 0), dh.get(2, 0)
            page = _decompress(codec, body, usize)
            p = 0
            if max_def:
                (ln,) = struct.unpack_from("<I", page, 0)
                defs = _hybrid(page, 4, 4 + ln, max_def.bit_length(), n)
                p = 4 + ln
            else:
                defs = [0] * n
        elif ptype_page == 3:  # data page v2: [rep levels][def levels] uncompressed, then values
            dh = header.get(8) or {}
            n, enc = dh.get(1, 0), dh.get(4, 0)
            dlen, rlen = dh.get(5, 0), dh.get(6, 0)
            if rlen:
                raise ParquetError("repeated columns are not supported")
            defs = _hybrid(body, 0, dlen, max_def.bit_length(), n) if max_def else [0] * n
            rest = body[dlen:]
            page = _decompress(codec, rest, usize - dlen) if dh.get(7, True) else rest
            p = 0
        else:
            continue  # index pages and anything else carry no values
        present = sum(1 for d in defs if d == max_def)
        if enc == 0:
            vals = _plain(ptype, page, p, present)
        elif enc in (2, 8):
            if dictionary is None:
                raise ParquetError("dictionary-encoded page without a dictionary")
            width = page[p]
            idx = _hybrid(page, p + 1, len(page), width, present)
            vals = [dictionary[i] for i in idx]
        else:
            raise ParquetError(f"unsupported value encoding {enc}")
        it = iter(vals)
        values.extend(next(it) if d == max_def else None for d in defs)
    return values


def read_table(source: str | Path | bytes) -> dict[str, list]:
    """{column name: values} of a flat Parquet file (BYTE_ARRAY values as bytes; None for nulls).

    A malformed file raises ``ParquetError`` (a ``ValueError``), never a low-level decoding error."""
    data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    try:
        return _read_table(data)
    except ParquetError:
        raise
    except (struct.error, zlib.error, IndexError, KeyError, TypeError, AttributeError, ValueError,
            OverflowError, RecursionError) as e:  # offsets or lengths that point outside the file, bad codecs
        raise ParquetError(f"malformed Parquet file ({type(e).__name__}: {e})") from None


def _read_table(data: bytes) -> dict[str, list]:
    if len(data) < 12 or data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ParquetError("not a Parquet file (missing PAR1 magic)")
    (meta_len,) = struct.unpack_from("<I", data, len(data) - 8)
    meta = _Thrift(data, len(data) - 8 - meta_len).struct()
    schema = meta.get(2) or []
    if not schema:
        raise ParquetError("Parquet file without a schema")
    leaves = schema[1:]
    if any(el.get(5) for el in leaves):
        raise ParquetError("nested columns are not supported")
    info = {}
    for el in leaves:
        name = el[4].decode("utf-8")
        rep = el.get(3, 0)
        if rep == 2:
            raise ParquetError(f"column {name!r} is repeated; not supported")
        info[name] = (el.get(1), 1 if rep == 1 else 0)
    out: dict[str, list] = {name: [] for name in info}
    for rg in meta.get(4) or []:
        for chunk in rg.get(1) or []:
            cm = chunk.get(3) or {}
            name = b".".join(cm.get(3) or []).decode("utf-8")
            if name not in info:
                raise ParquetError(f"column {name!r} is not in the schema")
            ptype, max_def = info[name]
            out[name].extend(_column(data, cm, ptype, max_def))
    return out
