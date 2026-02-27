"""
BPE Tokenizer implementation compatible with GPT-2 / tiktoken.
"""

from __future__ import annotations

import regex as re
from typing import Iterator
# 【新增】：我们需要引用 common 里的映射函数
try:
    from .common import gpt2_bytes_to_unicode
except ImportError:
    from common import gpt2_bytes_to_unicode

class Tokenizer:
    """
    A BPE (Byte Pair Encoding) tokenizer compatible with GPT-2.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Mapping from token ID to bytes
            merges: List of BPE merge pairs (bytes, bytes)
            special_tokens: List of special token strings
        """
        # Load the GPT-2 byte-to-unicode mapping
        # 【新增】：加载 GPT-2 的 字节->Unicode 映射表
        self.byte_encoder = gpt2_bytes_to_unicode()
        # 创建反向映射 Unicode -> 字节
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Transform vocab (int->bytes) to internal vocab (int->unicode_string)
        # We need this because _encode_chunk converts input text into these 
        # unicode characters, so the vocab keys must match.
        self.vocab = {}
        for token_id, token_bytes in vocab.items():
            # Map raw bytes to GPT-2 unicode string representation
            # This ensures that lookups in inverse_vocab work correctly
            token_str = "".join(self.byte_encoder[b] for b in token_bytes)
            self.vocab[token_id] = token_str

        # id -> bytes (now id -> str internally)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}  # bytes -> id (also used as rank)
        
        self.merges = merges
        # Note: We use inverse_vocab for BPE ranking, not the merges list.
        # In GPT-2/tiktoken, the token ID serves as the rank - lower ID = higher priority.
        # This is different from naive BPE which uses merge order.
        
        # Handle special tokens
        self.special_tokens = special_tokens or []
        # Sort special tokens by length (descending) for longest-match-first
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        
        # Build special token to ID mapping
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            # 【注意】：这里我们尝试查找 raw bytes，如果词表是 GPT-2 格式，
            # 可能需要额外的转换。
            # We map the special token bytes to the internal unicode representation
            token_mapped = "".join(self.byte_encoder[b] for b in token_bytes)
            
            if token_mapped in self.inverse_vocab:
                self.special_token_ids[token] = self.inverse_vocab[token_mapped]
        
        # GPT-2 regex pattern for pre-tokenization
        # This splits text into chunks that are tokenized independently
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )
        
    def _get_pairs(self, tokens: list[str]) -> set[tuple[str, str]]:
        """Get all adjacent pairs of tokens."""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def _bpe(self, token_parts: list[str]) -> list[str]:
        """
        Apply BPE to a single token (sequence of bytes).
        Returns a list of merged byte sequences.
        
        Uses vocab ranks (token IDs) to determine merge priority.
        Lower token ID = higher priority (more common/earlier merge).
        
        Algorithm:
            1. Start with individual bytes as tokens
            2. While there are pairs that can be merged:
               a. Find the pair whose merged result has the lowest vocab rank
               b. Merge all occurrences of that pair
            3. Return final token list
        """
        # Start with individual bytes
        # 【修改】：不再强制转 bytes，而是使用传入的 token_parts (可能是 Unicode 字符列表)
        tokens = list(token_parts)
        
        if len(tokens) <= 1:
            return tokens
        
        while True:
            # 1. 获取当前所有的相邻 token 对 (Get all adjacent pairs)
            pairs = self._get_pairs(tokens)
            if not pairs:
                break # 如果无法找到相邻对（例如长度降到1），则退出循环
                
            # 2. 找到在词表中 rank 最小的字节对 (Find pair with the lowest vocab rank)
            best_pair = None
            min_rank = float('inf')
            
            for pair in pairs:
                # 【修改】：在 GPT-2 中，合并是字符串拼接
                merged = pair[0] + pair[1]
                # 仅当合并后的字节存在于我们的词表（inverse_vocab）时才考虑
                if merged in self.inverse_vocab:
                    rank = self.inverse_vocab[merged]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair
                        
            # 如果没有找到任何可以合并的对，即当前的对都不在词表规则里，合并结束
            if best_pair is None:
                break
                
            # 3. 将所有出现的 best_pair 融合为新的 token (Merge all occurrences)
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 检查当前 token 与下一个 token 是否构成了最佳组合
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2 # 合并后，跳过下一个 token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            # 更新 tokens 为合并后的新序列，继续下一次循环
            tokens = new_tokens
            
        return tokens

    def _split_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text by special tokens, preserving them.
        Returns list of (substring, is_special) tuples.
        """
        if not self.special_tokens_sorted:
            return [(text, False)] if text else []
        
        result = []
        remaining = text
        
        while remaining:
            # Find the earliest occurring special token
            earliest_pos = len(remaining)
            earliest_token = None
            
            for special in self.special_tokens_sorted:
                pos = remaining.find(special)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = special
            
            if earliest_token is None:
                # No special token found, add remaining text
                if remaining:
                    result.append((remaining, False))
                break
            else:
                # Add text before the special token
                if earliest_pos > 0:
                    result.append((remaining[:earliest_pos], False))
                # Add the special token
                result.append((earliest_token, True))
                remaining = remaining[earliest_pos + len(earliest_token):]
        
        return result

    def _encode_chunk(self, text: str) -> list[int]:
        """
        Encode a text chunk (without special tokens) to token IDs.
        
        Algorithm:
            1. Use regex pattern (self.pat) to split text into pre-tokens
            2. For each pre-token:
               a. Convert to bytes
               b. Apply BPE to get list of byte sequences
               c. Convert each byte sequence to token ID using inverse_vocab
               d. Handle unknown tokens by falling back to individual bytes
        """
        if not text:
            return []
        
        ids = []
        # 1. 使用正则表达式切分文本获取 pre-tokens (Split text into pre-tokens)
        for match in self.pat.finditer(text):
            # 2a. 转换为字节形式 (Convert to bytes)
            token_bytes = match.group().encode("utf-8")
            
            # 【新增】：将 UTF-8 字节映射为 GPT-2 特有的 Unicode 字符列表
            # 这是修复 KeyError 的关键！
            token_chars = [self.byte_encoder[b] for b in token_bytes]
            
            # 2b. 执行底层的 BPE 算法 (Apply BPE)
            # 此时传入的是 Unicode 字符列表，而不是原始字节
            bpe_tokens = self._bpe(token_chars)
            
            # 2c & 2d. 转换为 Token ID (Convert to token ID)
            for bpe_token in bpe_tokens:
                if bpe_token in self.inverse_vocab:
                    ids.append(self.inverse_vocab[bpe_token])
                else:
                    # 极端边界情况：如果某个 token 不在词表中，将其拆散为单个字符再查表
                    # Fallback to individual chars if an unknown merged token appears
                    # Since we mapped all 256 bytes, individual chars should exist in vocab
                    for char in bpe_token:
                        if char in self.inverse_vocab:
                            ids.append(self.inverse_vocab[char])
                        # 注意：理论上 byte_encoder 覆盖了所有 256 个字节，所以单个 char 一定在词表中
                        
        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode a string to a list of token IDs.
        
        Args:
            text: Input string to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        ids = []
        
        # Split by special tokens first
        parts = self._split_with_special_tokens(text)
        
        for part, is_special in parts:
            if is_special:
                # Add special token ID
                if part in self.special_token_ids:
                    ids.append(self.special_token_ids[part])
            else:
                # Encode regular text
                ids.extend(self._encode_chunk(part))
        
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs to a string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        
        Algorithm:
            1. For each token_id, look up corresponding bytes in self.vocab
            2. Concatenate all byte chunks
            3. Decode as UTF-8 with errors="replace"
        """
        if not ids:
            return ""
        
        # 【修改】：因为 Vocab 里现在存的是 GPT-2 Unicode 字符串，
        # 我们先拼字符串，再通过 byte_decoder 转回字节
        text_chunks = []
        
        # 1. 遍历每个 token_id 去查词表
        for token_id in ids:
            if token_id in self.vocab:
                text_chunks.append(self.vocab[token_id])
                
        # 2. 拼接所有 Unicode 字符
        joined_text = "".join(text_chunks)
        
        # 3. 将 GPT-2 Unicode 字符映射回原始字节
        byte_data = bytearray()
        for char in joined_text:
            if char in self.byte_decoder:
                byte_data.append(self.byte_decoder[char])
            else:
                # 理论上不应该发生，除非 vocab 混入了奇怪的东西
                pass
                
        # 4. 解码为 UTF-8 字符串
        # Decode as UTF-8 with errors="replace"
        return byte_data.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        Yields token IDs one at a time without loading entire input into memory.
        
        Args:
            iterable: An iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        # Buffer for handling text that spans multiple lines
        buffer = ""
        
        for chunk in iterable:
            buffer += chunk
            
            # Process complete portions, keeping potential partial special tokens
            # Find the last safe split point
            safe_end = self._find_safe_split_point(buffer)
            
            if safe_end > 0:
                to_process = buffer[:safe_end]
                buffer = buffer[safe_end:]
                
                for token_id in self.encode(to_process):
                    yield token_id
        
        # Process remaining buffer
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def _find_safe_split_point(self, text: str) -> int:
        """
        Find a safe point to split text for streaming encoding.
        We need to be careful not to split in the middle of:
        1. A potential special token
        2. A whitespace sequence (to preserve tokens like '\\n\\n')
        """
        if not text:
            return 0
        
        # Check if any special token could be starting at the end
        max_special_len = max((len(s) for s in self.special_tokens), default=0)
        
        # We need to keep at least max_special_len - 1 characters in buffer
        # to avoid splitting a special token
        min_keep = max_special_len - 1 if max_special_len > 0 else 0
        
        if len(text) <= min_keep:
            return 0
        
        safe_end = len(text)
        
        # Check for partial special token matches at the end
        for special in self.special_tokens:
            # Check if any prefix of special token matches end of text
            for prefix_len in range(1, len(special)):
                prefix = special[:prefix_len]
                if text.endswith(prefix):
                    safe_end = min(safe_end, len(text) - prefix_len)
        
        # Don't split in the middle of trailing whitespace
        # This prevents breaking up tokens like '\n\n'
        if safe_end > 0:
            # Find the last non-whitespace character
            last_non_ws = safe_end - 1
            while last_non_ws >= 0 and text[last_non_ws].isspace():
                last_non_ws -= 1
            
            # If there's trailing whitespace, don't include it in this chunk
            # unless the entire text is whitespace
            if last_non_ws >= 0 and last_non_ws < safe_end - 1:
                safe_end = last_non_ws + 1
        
        return safe_end


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Create a tokenizer from vocabulary and merge rules.
    
    Args:
        vocab: Mapping from token ID to bytes
        merges: List of BPE merge pairs
        special_tokens: Optional list of special token strings
        
    Returns:
        Tokenizer instance
    """
    return Tokenizer(vocab, merges, special_tokens)