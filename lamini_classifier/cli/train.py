from lamini import LaminiClassifier

import argparse

from pprint import pprint

import pandas as pd


def main():
    """This is a program that trains a classifier using the LaminiClassifier class.

    LaminiClassifier is a powerful classifier that uses a large language model to classify data.

    It has the ability to define classes, each using a prompt, and then classify data based on that prompt.

    It can also be trained on examples of data for each class, and then classify data based on that training.

    """

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Parse the class names and prompts using the format "class_name:prompt"
    parser.add_argument(
        "--class",
        type=str,
        nargs="+",
        action="extend",
        help="The classes to use for classification, in the format 'class_name:prompt'.",
        default=[],
    )

    # Parse the training data from a csv file with two columns, class_name and data
    parser.add_argument(
        "--data",
        type=str,
        help="CSV filepath to the training data to use for classification, with the columns class_name and data.",
        default="data.csv",
    )

    # Parse the path to save the model to
    parser.add_argument(
        "--save",
        type=str,
        help="The path to save the model to.",
        default="models/model.lamini",
    )

    # Parse verbose mode
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print verbose output.",
        default=False,
    )

    # conert the arguments to a dictionary
    args = vars(parser.parse_args())

    # Create a new classifier
    classifier = LaminiClassifier()

    # Train the classifier on the classes
    classes = {}

    for class_prompt in args["class"]:
        class_name, prompt = class_prompt.split(":")
        assert class_name not in classes, f"Class name '{class_name}' already exists."
        classes[class_name] = prompt

    if len(classes) == 0:
        classes = get_default_classes()

    # Train the classifier on the training data
    data_df = pd.read_csv(args["data"])
    class_names = classes.keys()
    for _, row in data_df.iterrows():
        if row["class_name"] in class_names:
            # import pdb; pdb.set_trace()
            classifier.add_data_to_class(row["class_name"], [row["data"]])
        else:
            print(f"WARNING ------ Class name '{row['class_name']}' not found in classes, skipping.")

    if args["verbose"]:
        pprint("Training on classes:")
        pprint(classes)

    classifier.prompt_train(classes)

    if args["verbose"]:
        pprint(classifier.get_data())

    # Save the classifier to the path
    classifier.save(args["save"])


def get_default_classes():
    """Returns a dictionary of default classes to use for training.

    The default classes are:
        - "search"
        - "order"
        - "noop"
    """

    print("WARNING ------ No classes or data were specified, using default search, order, noop tools (classes).")

    return {
        "search": "User wants to get an answer about the food delivery app that is available in the FAQ pages of this app. This includes questions about their deliveries, payment, available grocery stores, shoppers, fees, and the app overall.",
        "order": "User wants to order items, i.e. buy an item and place it into the cart. This includes any indication of wanting to checkout, or add to or modify the cart. It includes mentioning specific items, or general turn of phrase like 'I want to buy something'.",
        "noop": "User didn't specify a tool, i.e. they didn't say they wanted to search or order. The ask is totally irrelevant to the delivery service app.",
    }

main()
