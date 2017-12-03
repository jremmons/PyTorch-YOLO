#!/usr/bin/env python

data_file_suffix_set = ("train", "val")


def convert_data_to_dict(data_file_suffix):
    data_dict = {}
    if data_file_suffix not in data_file_suffix_set:
        return data_dict

    return data_dict


def get_train_valid_dict():
    train_dict = convert_data_to_dict("train")
    valid_dict = convert_data_to_dict("valid")
    return train_dict, valid_dict


def main():
    train_dict, valid_dict = get_train_valid_dict()
    print("Train data dictionary:")
    print(train_dict)

    print("Valid data dictionary:")
    print(valid_dict)


if __name__ == "__main__":
    main()
