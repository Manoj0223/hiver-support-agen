from src.data_loader import (
    load_dataset,
    inspect_dataset
)


def main():

    print("=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 60)

    try:

        df = load_dataset()

        inspect_dataset(df)

    except Exception as e:

        print(
            "\nERROR:",
            e
        )


if __name__ == "__main__":
    main()
