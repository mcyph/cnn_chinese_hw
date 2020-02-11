def rem_dupes(L):
    S = set()
    return [x for x in L if x not in S and not S.add(x)]


def fast_rem_dupes(L):
    """
    Removes duplicates quickly but out-of-order
    """
    return list(set(L))


if __name__ == '__main__':
    from timeit import timeit
    L = ['a', 'a', 'b', 'c', 'd', 'e', 'e']

    print('rem_dupes:', timeit(lambda: rem_dupes(L), number=1000000))
    print('fast_rem_dupes:', timeit(lambda: fast_rem_dupes(L), number=1000000))
