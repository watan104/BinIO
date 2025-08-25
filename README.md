# BinIO

**BinIO**(BinaryIO) is a high-performance, header-only C++ library for binary I/O with optional SIMD acceleration (AVX2, SSE2, NEON).  
It provides a simple and safe API for reading and writing binary data, making it ideal for file parsers, networking, and serialization.

---

## Features
- **Header-only**: just `#include "Binary.hpp"`
- **Cross-platform**: works on Windows, Linux, macOS
- **SIMD accelerated** (AVX2, SSE2, NEON) for high performance
- **Reader**: easy sequential binary parsing
- **Writer**: efficient binary serialization
- Memory-aligned buffers and cache-friendly design

---

## 🚀 Example Usage

**Reader**:

```cpp
#include "Binary.hpp"
#include <iostream>
#include <vector>

int main() {
    // Example data (little-endian: 1, 2)
    std::vector<uint8_t> data = {0x01, 0x00, 0x02, 0x00};

    BinIO::Reader reader(data);

    uint16_t a = reader.read<uint16_t>();
    uint16_t b = reader.read<uint16_t>();

    std::cout << a << " " << b << std::endl; // Output: 1 2
}
```

**Writer**

```cpp
#include "Binary.hpp"
#include <iostream>

int main() {
    BinIO::Writer writer(16);

    writer.write<uint32_t>(12345);
    writer.write<float>(3.14f);

    const uint8_t* buffer = writer.get();

    std::cout << "First value: " << *reinterpret_cast<const uint32_t*>(buffer) << std::endl;
}
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Binary.hpp.git
```
2. ``#include "Binary.hpp"`` in your C++ files.

  
