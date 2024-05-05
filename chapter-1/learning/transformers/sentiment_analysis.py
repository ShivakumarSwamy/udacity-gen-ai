import transformers


def main():
    classifier = transformers.pipeline('sentiment-analysis')
    result = classifier([
        'I am happy today!',
        'I am not happy today!'
    ])

    print(result)



if __name__ == '__main__':
    main()
