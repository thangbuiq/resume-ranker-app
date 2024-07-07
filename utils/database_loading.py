import pandas as pd


# Define function to check for invalid skill lists (empty lists or lists with empty strings)
def is_invalid_skill(skill):
    return skill == "[]" or skill == "['']"


def load_data(path):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    # Filter DataFrame to drop rows with invalid skill lists in either 'skills' or 'clean_skills'
    df = df[
        ~df["skills"].apply(is_invalid_skill)
        & ~df["clean_skills"].apply(is_invalid_skill)
    ]

    # Drop rows with NaN values in specific critical columns if needed
    df = df.dropna(subset=["profile_picture", "Name", "position", "location"])

    labels_dict = {label: idx for idx, label in enumerate(df.category.unique())}
    df.category = df.category.apply(lambda x: labels_dict[x]).astype(int)

    df["name_candidate"] = df["Name"]

    # Gộp các cột thông tin thành một cột resume
    df["Resume"] = df.apply(
        lambda row: " ".join(
            [
                str(row["description"]) if pd.notna(row["description"]) else "",
                str(row["clean_skills"]) if pd.notna(row["clean_skills"]) else "",
                str(row["Experience"]) if pd.notna(row["Experience"]) else "",
            ]
        ),
        axis=1,
    )

    # Chỉ giữ lại các cột index, name_candidate và resume
    df = df[
        [
            "index",
            "name_candidate",
            "category",
            "Resume",
            "linkedin",
            "Experience",
            "description",
        ]
    ]

    return df
