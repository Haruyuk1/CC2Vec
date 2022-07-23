import javalang


def main():
    exmaple_fragment = 'System.out.println("Hello World!\n")'
    gen = javalang.tokenizer.tokenize(exmaple_fragment)

    for token in gen:
        print(f'type:{type(token)}, value:{token.value}')


    return


if __name__ == "__main__":
    main()