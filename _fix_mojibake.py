"""Repair the mojibake in DESIGN.md's diagram. The box-drawing
characters were double-encoded as UTF-8 then misread as latin-1,
producing sequences like 'Ã' followed by control bytes."""
from pathlib import Path

p = Path(r"D:\Work\rytp\DESIGN.md")
raw = p.read_bytes()

# The corruption pattern: a 3-byte sequence that, when re-decoded
# as latin-1, gives the original UTF-8 bytes. Easiest fix:
# read the bytes as latin-1 to recover a UTF-8 string, then
# decode that as UTF-8.
text = raw.decode("latin-1")

# But not every byte sequence is mojibake — only the ones where
# the re-decoded UTF-8 is a real box-drawing char. Replace only
# those.
import re

# Box-drawing range U+2500..U+257F. After the fix-up, these should
# appear as single chars in the resulting string.
fixed_chars = set(chr(c) for c in range(0x2500, 0x2580))

# Walk: wherever we have a sequence that decodes to a box-drawing
# char, replace it with that char. A simpler way: encode the
# latin-1-decoded string back to latin-1 to get the original bytes,
# then decode as UTF-8. The non-mojibake ASCII is identical under
# both encodings.
try:
    out = text.encode("latin-1").decode("utf-8")
    p.write_bytes(out.encode("utf-8"))
    print("OK")
except UnicodeDecodeError as e:
    # The text is not entirely mojibake-decodable; we'd need a
    # targeted fix. Bail and tell the user.
    print(f"FAIL: {e}")
    # Find the offset in the latin-1-decoded text
    bad_offset = e.start
    print(f"first bad offset: {bad_offset}")
    print(f"context: {text[max(0, bad_offset-20):bad_offset+20]!r}")
