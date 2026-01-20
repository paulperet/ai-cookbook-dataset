

# Gemini API: Basic evaluation

Gemini API's Python SDK can be used for various forms of evaluation, including:
- Providing feedback based on selected criteria
- Comparing multiple texts
- Assigning grades or confidence scores
- Identifying weak areas

Below is an example of using the LLM to enhance text quality through feedback and grading.


```
%pip install -U -q "google-genai>=1.7.0"
```


```
from google import genai
from google.genai import types
from IPython.display import Markdown
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Example

Start by defining some system instructions for this problem. For demonstration purposes, the use case involves prompting the model to write an essay with some mistakes. Remember that for generation tasks like writing an essay, you can change the temperature of the model to get more creative answers.


```
student_system_prompt = """
    You're a college student. Your job is to write an essay riddled with common mistakes and a few major ones.
    The essay should have mistakes regarding clarity, grammar, argumentation, and vocabulary.
    Ensure your essay includes a clear thesis statement. You should write only an essay, so do not include any notes.
"""

MODEL_ID="gemini-2.5-flash" # @param ["gemini-2.5-flash-lite","gemini-2.5-flash","gemini-2.5-pro","gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

student_response = client.models.generate_content(
    model=MODEL_ID,
    contents='Write an essay about the benefits of reading.',
    config=types.GenerateContentConfig(
        system_instruction=student_system_prompt
    ),
)

essay = student_response.text
Markdown(essay)
```

Reading is like, a really good thing. It’s beneficial, you know? Like, a lot. That's what I'm gonna talk about. So, like, my thesis is that reading is really important for a bunch of reasons that I will now tell you.

First off, reading makes you smart. I read this somewhere, I forget where. Probably on the internet. But, like, think about it. When you read, you learn new words. And new words are good because you can use them to impress people. The more words you know, the smarter you seem. Its like a secret code to being smart, I think. My English teacher is always saying, “use a thesaurus!” So, like, I do. Sometimes. Its useful. Plus, reading exposes you to new ideas, and new ideas are good for…stuff.

Secondly, reading books is way better than watching TV. TV rots your brain. Everyone knows that. I mean, have you ever met someone who watches TV all day and is, like, a genius? No, right? Exactly. But books, books are different. Books make you think, even if the stuff you read doesn't make much sense. I read this book last week, something with like, a red cover, and it didn't make any sense. But I know it was making me smarter, because I kept having to go back and try to understand it. Therefore, books good, TV bad.

Finally, reading can help you with writing. The more you read, the more you pick up on how other people write, their, like, style and stuff. So when it comes time for you to write something, like this essay even, it will be easier. It's like you’re absorbing their talent through your eyeballs. I think it's osmosis or something. And also, it's because you learn all these new words, that I mentioned before. I always learn a whole bunch of new words whenever I am reading. Its just natural, you know.

In conclusion, reading is really, really good. It makes you smart, its better than watching TV, and it helps you with writing. So everyone should read more. Seriously, go read a book. I did, and look at me, writing this essay! Like, I am super benefiting from reading right now. You should to. Also, I hope I get a good grade on this essay.


```
teacher_system_prompt = """
    As a teacher, you are tasked with grading students' essays.
    Please follow these instructions for evaluation:

    1. Evaluate the essay on a scale of 1-5 based on the following criteria:
       - Thesis statement,
       - Clarity and precision of language,
       - Grammar and punctuation,
       - Argumentation

    2. Write a corrected version of the essay, addressing any identified issues
       in the original submission. Point out what changes were made.
"""
teacher_response = client.models.generate_content(
    model=MODEL_ID,
    contents=essay,
    config=types.GenerateContentConfig(
        system_instruction=teacher_system_prompt
    ),
)


Markdown(teacher_response.text)
```

Okay, here's a breakdown of your essay, along with a corrected version.

**Evaluation:**

*   **Thesis Statement (2/5):** The thesis is present but lacks sophistication and clarity. It's too conversational and doesn't precisely outline the specific arguments.
*   **Clarity and Precision of Language (2/5):** The language is overly informal, repetitive ("like," "you know"), and lacks precision. The writing could benefit from more specific examples and a more academic tone.
*   **Grammar and Punctuation (3/5):** There are noticeable errors in grammar and punctuation, including incorrect use of "its" vs. "it's," run-on sentences, and comma splices.
*   **Argumentation (2/5):** The arguments are simplistic and lack strong evidence. The connections between reading and intelligence, writing ability, etc., are not fully developed or supported with credible sources.

**Corrected Essay:**

Reading offers multifaceted benefits that extend beyond mere entertainment. Its importance is underscored by its capacity to enhance cognitive abilities, provide a superior alternative to passive screen time, and cultivate effective writing skills.

Firstly, reading significantly contributes to intellectual development. Exposure to a wide range of texts expands vocabulary and enhances comprehension. As one encounters new words and concepts within various contexts, they are integrated into one's cognitive framework, facilitating more nuanced and articulate communication. As such, reading becomes a powerful tool for self-improvement and intellectual growth. Furthermore, reading exposes individuals to diverse perspectives and ideas, which stimulates critical thinking and broadens one's understanding of the world.

Secondly, engaging with literature offers a more enriching experience compared to passive forms of media consumption like television. While television often presents information in a simplified and visually driven format, reading requires active engagement and interpretation. This active engagement fosters concentration, analytical skills, and imagination, leading to deeper cognitive processing. Moreover, the ability to select books allows for a tailored and purposeful learning experience, unlike the often random and unfiltered nature of television programming.

Finally, reading plays a crucial role in the development of writing proficiency. By immersing themselves in the works of skilled authors, readers subconsciously absorb stylistic elements, narrative structures, and effective rhetorical techniques. This implicit learning process enhances one's ability to articulate thoughts clearly, construct compelling arguments, and employ language effectively. Furthermore, the increased vocabulary acquired through reading directly translates into more precise and varied writing.

In conclusion, reading offers a wealth of cognitive, intellectual, and creative advantages. Its capacity to enhance intellect, surpass the limitations of passive media, and foster effective writing skills underscores its importance in personal and academic development. Therefore, individuals are encouraged to embrace reading as a means of lifelong learning and intellectual enrichment.

**Explanation of Changes:**

*   **Thesis Statement:** The original thesis was very informal. The corrected version is more formal and specific, clearly stating the three main benefits.
*   **Language:** All instances of "like," "you know," and other informal expressions were removed. The language was made more precise and academic.
*   **Grammar and Punctuation:** Grammatical errors were corrected (e.g., "its" changed to "it's"). Run-on sentences were split, and commas were added or removed for clarity.
*   **Argumentation:** The arguments were strengthened by providing more specific explanations of how reading leads to intelligence, better writing, etc. The vague references to "reading it somewhere" were removed. While still lacking direct citations, the corrected version presents more logical reasoning.
*   **Structure:** The essay was structured with clear topic sentences for each paragraph and a more formal concluding paragraph.
*   **Tone:** The overall tone was shifted from conversational to academic.

## Next steps

Be sure to explore other examples of prompting in the repository. Try writing your own prompts for evaluating texts.