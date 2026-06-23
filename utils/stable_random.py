import hashlib


def stable_randbelow(seed: str, upper: int, counter: int) -> int:
    if upper <= 0:
        raise ValueError('upper must be positive')

    nbits = upper.bit_length()
    nbytes = (nbits + 7) // 8
    limit = (1 << (8 * nbytes)) - ((1 << (8 * nbytes)) % upper)

    nonce = 0
    while True:
        msg = f'{seed}:{counter}:{nonce}'.encode('utf-8')
        digest = hashlib.sha256(msg).digest()
        value = int.from_bytes(digest[:nbytes], 'big')
        if value < limit:
            return value % upper
        nonce += 1


def stable_shuffle(values, seed: str):
    shuffled = list(values)
    counter = 0

    for index in range(len(shuffled) - 1, 0, -1):
        swap_index = stable_randbelow(seed, index + 1, counter)
        shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
        counter += 1

    return shuffled
