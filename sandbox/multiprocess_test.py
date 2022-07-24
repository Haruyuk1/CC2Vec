import concurrent.futures


def f2(s):
    return len(s)


def main():
    def f(s):
        return len(s)

    data = ["a", "b", "c"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        # results = pool.map(f, data) # hangs
        # results = pool.map(lambda d: len(d), data)  # hangs
        # results = pool.map(len, data)  # works
        results = pool.map(f2, data) # works

    print(list(results))


if __name__ == "__main__":
    main()