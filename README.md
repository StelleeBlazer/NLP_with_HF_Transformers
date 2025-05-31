<h1 align="center"> Natural Language Processing with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Muhammad Rahmananda Arief Wibisono

## My todo :

### 1. Sentiment Analysis

```python
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am doing this on a regular basis, baking a cake in the morning!")

Result:
[{'label': 'POSITIVE', 'score': 0.9959210157394409}]

Analysis:
The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.

2. Topic Classification

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Cats are beloved domestic companions known for their independence and agility. These fascinating creatures exhibit a range of behaviors, from playful pouncing to peaceful purring. With their sleek fur, captivating eyes, and mysterious charm, cats have captivated humans for centuries, becoming cherished members of countless households worldwide.",
    candidate_labels=["science", "pet", "machine learning"],
)

Result:
Device set to use cuda:0
{'sequence': 'Cats are beloved domestic companions known for their independence and agility. These fascinating creatures exhibit a range of behaviors, from playful pouncing to peaceful purring. With their sleek fur, captivating eyes, and mysterious charm, cats have captivated humans for centuries, becoming cherished members of countless households worldwide.',
 'labels': ['pet', 'machine learning', 'science'],
 'scores': [0.9174826145172119, 0.048576705157756805, 0.03394068405032158]}

Analysis:
The zero-shot classifier correctly identifies "pet" as the most relevant label, with a high confidence score. This shows the model's strong ability to associate descriptive context with predefined categories, even without task-specific fine-tuning or training on the input text.

3. Text Generation
# TODO :
text_generator = pipeline("text-generation", model="gpt2")
text_generator("The future of artificial intelligence is", max_length=50, num_return_sequences=2)

Result:
Device set to use cuda:0
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Both `max_new_tokens` (=256) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[{'generated_text': "The future of artificial intelligence is almost certainly on the horizon. We can see it now, with deep learning tools, in a few years. The future is definitely on the horizon. We can see it now, with deep learning tools, in a few years. The next big thing is AI. No one can really tell you how AI is going to transform the world, but there are real opportunities.\n\nIn the next week, I will be speaking at the World Economic Forum in Davos, Switzerland. I'm going to talk about the future of AI and the technologies that are coming. I'm going to talk about the future of AI and the technologies that are coming. I'm going to talk about the future of AI and the technologies that are coming. I'm going to talk about the future of AI and the technologies that are coming.\n\nI'm here to talk about the future of AI and the technologies that are coming. I'm here to talk about the future of AI and the technologies that are coming. I'm here to talk about the future of AI and the technologies that are coming. I'm here to talk about the future of AI and the technologies that are coming.\n\nI'm here to talk about the future of AI and the technologies that are coming. I'm here"},
 {'generated_text': 'The future of artificial intelligence is under threat from the advances of artificial intelligence. The next step in that direction is artificial intelligence that is designed to solve difficult problems and to make decisions.\n\nThe future of artificial intelligence is under threat from the advances of artificial intelligence. The next step in that direction is artificial intelligence that is designed to solve difficult problems and to make decisions.\n\nThe ultimate goal of AI is to make decisions and to make them intelligible — perhaps as soon as possible. That is why we have such a high level of confidence in these systems. They are well equipped to make the right decisions in a very short time. But as soon as we have all the technologies, we will have to make the most efficient decisions.\n\nFor example, the most efficient way of making decisions is to use machines. The next step is to get rid of machines, because it\'s not as easy to do it, and that\'s not the way to go.\n\nBut there are three steps to make decisions. First, we have to get rid of the "bots." The AI is designed to make the right decisions. But the next step is to get rid of the "bots." The AI is designed to make the right decisions. But the next step is to get rid of the "bots'}]

Analysis:
In the context of my case study relating to AI storytellers for folklore preservation, the text generation capabilities of GPT-2 can be used to generate natural and engaging story narratives. Although the model has not been specifically trained for the local cultural context, it shows the potential to creatively extend the story from a single opening sentence, making it useful as an initial foundation for automated storytelling.

4. Named Entity Recognition (NER)
# TODO :
ner = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
ner("Emmanuel Macron est le président de la République française et il travaille à l'Élysée à Paris.")

Result:
Device set to use cuda:0
[{'entity_group': 'PER',
  'score': np.float32(0.99842584),
  'word': 'Emmanuel Macron',
  'start': 0,
  'end': 15},
 {'entity_group': 'PER',
  'score': np.float32(0.81151164),
  'word': 'président de la République',
  'start': 22,
  'end': 49},
 {'entity_group': 'LOC',
  'score': np.float32(0.9542574),
  'word': 'Élysée',
  'start': 80,
  'end': 86},
 {'entity_group': 'LOC',
  'score': np.float32(0.994638),
  'word': 'Paris',
  'start': 88,
  'end': 94}]

Analysis:
The CamemBERT model managed to accurately recognize entities in French sentences, including names of people, organizations, and locations. In my case study, this NER feature is important for marking entities in folklore such as legendary characters, places, or kingdoms. If adapted to Indonesian or local languages, NER can be used to automatically annotate important elements in a story.

Exercise 5 - Question Answering
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

question = "Where does Lionel Messi play?"
context = "Lionel Messi is a professional footballer who plays for Inter Miami in Major League Soccer."

qa_model(question=question, context=context)

Result:
Device set to use cuda:0
{'score': 0.9579006433486938, 'start': 56, 'end': 67, 'answer': 'Inter Miami'}

Analysis:
This QA model is capable of answering context-based questions with high accuracy. In my case study, this capability can be used to create an automated question-and-answer feature on stories, for example answering “Who is the main character in this story?” or “Where did this story take place?” This will be very useful for user interaction with educational story content.

Exercise 6 - Text Summarization
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer(
    """
Data science is a field that uses various methods, algorithms, and systems to extract insights and knowledge from data.
It combines aspects of statistics, computer science, and domain expertise. Data scientists often deal with large datasets,
clean and process them, and apply machine learning models to make predictions or uncover hidden patterns.
The field is crucial in industries like healthcare, finance, marketing, and more. With the rise of big data,
data science has become increasingly important in helping businesses make data-driven decisions.
"""
)

Result:
Device set to use cuda:0
Your max_length is set to 142, but your input_length is only 112. Since this is a summarization task, where outputs shorter than the input are typically wanted, you might consider decreasing max_length manually, e.g. summarizer('...', max_length=56)
[{'summary_text': ' Data scientists often deal with large datasets, clean and process them, and apply machine learning models to make predictions or uncover hidden patterns . The field is crucial in industries like healthcare, finance, marketing and more . With the rise of big data, Data Science has become increasingly important in helping businesses make data-driven decisions .'}]

Analysis:
This summarization model is effective for summarizing long texts into a more concise essence. In folklore preservation projects, this model can be utilized to create highlights of long stories, so that users can read the summary first before listening to the full story.

Exercise 7 - Translation
# TODO :
translator_en_de = pipeline("translation_en_to_de")

translator_en_de("I really enjoy learning new languages and exploring different cultures.")

Result:
Device set to use cuda:0
[{'translation_text': 'Ich lerne gerne neue Sprachen und erkunde verschiedene Kulturen.'}]

Analysis:
Although this model translates from English to German, such capabilities are important in my case study to expand the reach of folklore to a global audience. In the future, translating stories into different languages (including local) can facilitate cross-cultural distribution and preservation.
