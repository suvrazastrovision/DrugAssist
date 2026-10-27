# Tokenizer test asset

`cl100k_base.tiktoken` is the real vocabulary used by the chunking tests. Tests
load this file explicitly and never download it or consult a user's tokenizer cache.

- Source: https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
- SHA-256: `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`
- Upstream: OpenAI's tiktoken, https://github.com/openai/tiktoken
- License: MIT; see the adjacent `TIKTOKEN_LICENSE.txt`.

When changing encodings, add the corresponding immutable asset, verify its upstream
checksum, and update the test loader. Do not replace real tokenization with a mock.
