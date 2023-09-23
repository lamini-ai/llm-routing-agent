
# Routing agent that uses tools with [Lamini](https://lamini.ai) & Llama 2

Train a new routing agent that uses tools with just prompts.

```bash
./train.sh
```

You can specify your own tools (and descriptions of them using prompts) and optionally add training data to each class, by just adding a flag that specifies the path to a CSV file with columns "class_name" for the tool and "data" for an example user query.

```bash
 ./train.sh
 --class "search: SEARCH_TOOL_DESCRIPTION" 
 --class "order: ORDERING_TOOL_DESCRIPTION" 
 --class "noop: NO_TOOL_DESCRIPTION"
 --data CSV_DATA_FILEPATH
```

Then, route a new user request:
```bash
./classify.sh 'I want to buy three of these organic bananas now'
```

```python
{
  'data': 'I want to buy three of these organic bananas now',
  'prediction': 'order',
  'probabilities': array([0.37626405, 0.4238198 , 0.19991615])
 }
```

Run your LLM agent on multiple examples at once:
```bash
./classify.sh --data "I'd like to find a plumber" --data "who is the best person to ping for delivery issues?" --data "I want to buy three of these organic bananas now"
```

```python
{'data': "I'd like to find a plumber",
 'prediction': 'noop',
 'probabilities': array([0.34477281, 0.25205545, 0.40317174])}
{'data': 'who is the best person to ping for delivery issues?',
 'prediction': 'search',
 'probabilities': array([0.67681697, 0.13177853, 0.1914045 ])}
{'data': 'I want to buy three of these organic bananas now',
 'prediction': 'order',
 'probabilities': array([0.37626405, 0.4238198 , 0.19991615])}
 ```

For example, here is a routing agent for a food delivery app that uses the tools {search, order, noop} trained using prompts.

Search prompt:

```
User wants to get an answer about the food delivery app that is available in the FAQ pages of this app. This includes questions about their deliveries, payment, available grocery stores, shoppers, fees, and the app overall.
```

Order prompt:
```
User wants to order items, i.e. buy an item and place it into the cart. This includes any indication of wanting to checkout, or add to or modify the cart. It includes mentioning specific items, or general turn of phrase like 'I want to buy something'.
```

Noop prompt:
```
User didn't specify a tool, i.e. they didn't say they wanted to search or order. The ask is totally irrelevant to the delivery service app."
```

# Installation

Clone this repo, and run the `train.sh` or `classify.sh` command line tools.  

Requires docker: https://docs.docker.com/get-docker 

Setup your lamini keys (free): https://lamini-ai.github.io/auth/

`git clone git@github.com:lamini-ai/llm_routing_agent.git`

`cd llm_routing_agent`

Train a new classifier.

```
./train.sh --help

usage: train.py [-h] [--class CLASS [CLASS ...]] [--train TRAIN [TRAIN ...]] [--save SAVE] [-v]

options:
  -h, --help            show this help message and exit
  --class CLASS [CLASS ...]
                        The classes to use for classification, in the format 'class_name:prompt'.
  --train TRAIN [TRAIN ...]
                        The training data to use for classification, in the format 'class_name:data'.
  --save SAVE           The path to save the model to.
  -v, --verbose         Whether to print verbose output.

```

Classify your data.

```
./classify.sh --help

usage: classify.py [-h] [--data DATA [DATA ...]] [--load LOAD] [-v] [classify ...]

positional arguments:
  classify              The data to classify.

options:
  -h, --help            show this help message and exit
  --data DATA [DATA ...]
                        The training data to use for classification, any string.
  --load LOAD           The path to load the model from.
  -v, --verbose         Whether to print verbose output.

```

These command line scripts just call python inside of docker so you don't have to care about an environment.  

If you hate docker, you can also run this from python easily...


# Python Library

Install it
`pip install lamini`

Instantiate a classifier

```python
from lamini import LaminiClassifier

# Create a new classifier
classifier = LaminiClassifier()
```

Define classes using prompts

```python
classes = { "SOME_CLASS" : "SOME_PROMPT" }

classifier.prompt_train(classes)
```

Add some training examples (optional)

```python
data = ["example 1", "example 2"]
classifier.add_data_to_class("SOME_CLASS", data)

# Don't forget to train after adding data
classifier.train()
```

Classify your data

```python
# Classify the data
prediction = classifier.predict(data)

# Get the probabilities for each class
probabilities = classifier.predict_proba(data)
```

Save your model

```python
classifier.save("SOME_PATH")
```

Load your model
```python
classifier = LaminiClassifier.load(args["load"])
```

# FAQ

## How does it work?

The LLM routing agent converts your prompts about tools into a pile of data about those tools, using the Llama 2 LLM. It then finetunes another LLM to distinguish between each pile of data. This was forked from the [LLM Classifier](https://github.com/lamini-ai/llm_classifier/).

We use several specialized LLMs derived from Llama 2 to convert prompts into piles of training examples for each class.  The code for this is available
in the lamini python package if you want to look at it.  Working on open sourcing in an easier to read github page it when I'm not too distracted...

## Is this perfect?

No, this is a week night hackathon project, give us feedback and we will improve it.  Some known issues:

1. It doesn't use batching aggressively over classes, so training on many classes could be sped up by more than 100x.
2. We are refining the LLM example generators.  Send us any issues you find with your prompts adn we can improve these models.

## Why wouldn't I just use a normal classifier like BART, XGBoost, BERT, etc?

You don't need to label any data using the `LaminiClassifier`.  Labeling data sucks.

No fiddling with hyperparameters. Fiddle with prompts instead.  Hopefully english is easier than attention_dropout_pcts.

## Why wouldn't I just use a LLM directly?

A classifier always outputs a valid class.  An LLM might answer the question "Is this talking about an order" with "Well... that depends on ....".  Writing a parser sucks.

Added benefit: classifiers give you probabilities and can be calibrated: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/


