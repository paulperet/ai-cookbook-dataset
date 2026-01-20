# How to handle rate limits

When you call the OpenAI API repeatedly, you may encounter error messages that say `429: 'Too Many Requests'` or `RateLimitError`. These error messages come from exceeding the API's rate limits.

This guide shares tips for avoiding and handling rate limit errors.

To see an example script for throttling parallel requests to avoid rate limit errors, see [api_request_parallel_processor.py](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py).

## Why rate limits exist

Rate limits are a common practice for APIs, and they're put in place for a few different reasons.

- First, they help protect against abuse or misuse of the API. For example, a malicious actor could flood the API with requests in an attempt to overload it or cause disruptions in service. By setting rate limits, OpenAI can prevent this kind of activity.
- Second, rate limits help ensure that everyone has fair access to the API. If one person or organization makes an excessive number of requests, it could bog down the API for everyone else. By throttling the number of requests that a single user can make, OpenAI ensures that everyone has an opportunity to use the API without experiencing slowdowns.
- Lastly, rate limits can help OpenAI manage the aggregate load on its infrastructure. If requests to the API increase dramatically, it could tax the servers and cause performance issues. By setting rate limits, OpenAI can help maintain a smooth and consistent experience for all users.

Although hitting rate limits can be frustrating, rate limits exist to protect the reliable operation of the API for its users.

## Default rate limits

Your rate limit and spending limit (quota) are automatically adjusted based on a number of factors. As your usage of the OpenAI API goes up and you successfully pay the bill, we automatically increase your usage tier. You can find specific information regarding rate limits using the resources below.

### Other rate limit resources

Read more about OpenAI's rate limits in these other resources:

- [Guide: Rate limits](https://platform.openai.com/docs/guides/rate-limits?context=tier-free)
- [Help Center: Is API usage subject to any rate limits?](https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits)
- [Help Center: How can I solve 429: 'Too Many Requests' errors?](https://help.openai.com/en/articles/5955604-how-can-i-solve-429-too-many-requests-errors)

### Requesting a rate limit increase

To learn more about increasing your organization's usage tier and rate limit, visit your [Limits settings page](https://platform.openai.com/account/limits).


```python
import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

## Example rate limit error

A rate limit error will occur when API requests are sent too quickly. If using the OpenAI Python library, they will look something like:

```
RateLimitError: Rate limit reached for default-codex in organization org-{id} on requests per min. Limit: 20.000000 / min. Current: 24.000000 / min. Contact support@openai.com if you continue to have issues or if you’d like to request an increase.
```

Below is example code for triggering a rate limit error.


```python
# request a bunch of completions in a loop
for _ in range(100):
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10,
    )
```

[RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}]

## How to mitigate rate limit errors

### Retrying with exponential backoff

One easy way to mitigate rate limit errors is to automatically retry requests with a random exponential backoff. Retrying with exponential backoff means performing a short sleep when a rate limit error is hit, then retrying the unsuccessful request. If the request is still unsuccessful, the sleep length is increased and the process is repeated. This continues until the request is successful or until a maximum number of retries is reached.

This approach has many benefits:

- Automatic retries means you can recover from rate limit errors without crashes or missing data
- Exponential backoff means that your first retries can be tried quickly, while still benefiting from longer delays if your first few retries fail
- Adding random jitter to the delay helps retries from all hitting at the same time

Note that unsuccessful requests contribute to your per-minute limit, so continuously resending a request won’t work.

Below are a few example solutions.

#### Example #1: Using the Tenacity library

[Tenacity](https://tenacity.readthedocs.io/en/latest/) is an Apache 2.0 licensed general-purpose retrying library, written in Python, to simplify the task of adding retry behavior to just about anything.

To add exponential backoff to your requests, you can use the `tenacity.retry` [decorator](https://peps.python.org/pep-0318/). The following example uses the `tenacity.wait_random_exponential` function to add random exponential backoff to a request.

Note that the Tenacity library is a third-party tool, and OpenAI makes no guarantees about its reliability or security.


```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


completion_with_backoff(model="gpt-4o-mini", messages=[{"role": "user", "content": "Once upon a time,"}])
```

[ChatCompletion(id='chatcmpl-8PAu6anX2JxQdYmJRzps38R8u0ZBC', choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content='in a small village nestled among green fields and rolling hills, there lived a kind-hearted and curious young girl named Lily. Lily was known for her bright smile and infectious laughter, bringing joy to everyone around her.

One sunny morning, as Lily played in the meadows, she stumbled upon a mysterious book tucked away beneath a tall oak tree. Intrigued, she picked it up and dusted off its weathered cover to reveal intricate golden patterns. Without hesitation, she opened it, discovering that its pages were filled with magical tales and enchanting adventures.

Among the stories she found, one particularly caught her attention—a tale of a long-lost treasure hidden deep within a mysterious forest. Legend had it that whoever found this hidden treasure would be granted one wish, no matter how big or small. Excited by the prospect of finding such treasure and fulfilling her wildest dreams, Lily decided to embark on a thrilling journey to the forest.

Gathering her courage, Lily told her parents about the magical book and her quest to find the hidden treasure. Though concerned for their daughter's safety, they couldn't help but admire her spirit and determination. They hugged her tightly and blessed her with love and luck, promising to await her return.

Equipped with a map she found within the book, Lily ventured into the depths of the thick forest. The trees whispered tales of forgotten secrets, and the enchanted creatures hidden within watched her every step. But Lily remained undeterred, driven by her desire to discover what lay ahead.

Days turned into weeks as Lily traversed through dense foliage, crossed swift rivers, and climbed treacherous mountains. She encountered mystical beings who offered guidance and protection along her perilous journey. With their help, she overcame countless obstacles and grew braver with each passing day.

Finally, after what felt like an eternity, Lily reached the heart of the forest. There, beneath a jeweled waterfall, she found the long-lost treasure—a magnificent chest adorned with sparkling gemstones. Overwhelmed with excitement, she gently opened the chest to reveal a brilliant light that illuminated the forest.

Within the glow, a wise voice echoed, "You have proven your courage and pure heart, young Lily. Make your wish, and it shall be granted."

Lily thought deeply about her wish, realizing that her true treasure was the love and happiness she felt in her heart. Instead of making a wish for herself, she asked for the wellbeing and prosperity of her village, spreading joy and harmony to everyone living there.

As the light faded, Lily knew her quest was complete. She retraced her steps through the forest, returning home to find her village flourishing. Fields bloomed with vibrant flowers, and laughter filled the air.

The villagers greeted Lily with open arms, recognizing her selflessness and the magic she had brought into their lives. From that day forward, they told the tale of Lily's journey, celebrating her as a heroine who embodied the power of love, kindness, and the belief that true treasure lies within oneself.

And so, the story of Lily became an everlasting legend, inspiring generations to follow their dreams, be selfless, and find the true treasures that lie within their hearts.', role='assistant', function_call=None, tool_calls=None))], created=1701010806, model='gpt-3.5-turbo-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=641, prompt_tokens=12, total_tokens=653))]

#### Example #2: Using the backoff library

Another library that provides function decorators for backoff and retry is [backoff](https://pypi.org/project/backoff/).

Like Tenacity, the backoff library is a third-party tool, and OpenAI makes no guarantees about its reliability or security.


```python
import backoff  # for exponential backoff

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=6)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


completions_with_backoff(model="gpt-4o-mini", messages=[{"role": "user", "content": "Once upon a time,"}])

```

[ChatCompletion(id='chatcmpl-AqRiD3gF3q8VVs6w8jgba6FHGr0L5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="in a small village nestled between lush green hills and a shimmering lake, there lived a young girl named Elara. Elara had a curious spirit and a heart full of dreams. Every day, she would explore the woods surrounding her home, searching for hidden treasures and magical creatures.

One sunny afternoon, while wandering deeper into the forest than she ever had before, Elara stumbled upon a sparkling, crystal-clear pond. As she knelt down to take a closer look, she noticed a glimmering object at the bottom. It was a beautifully crafted key, shining with an otherworldly light. Without thinking twice, Elara reached into the cool water and retrieved the key, feeling a strange warmth envelop her.

Little did she know, this key was no ordinary key. It was said to unlock a secret door hidden in the heart of the forest, a door that led to a realm of wonder and adventure. Legends whispered of enchanted beings, ancient wisdom, and challenges that could only be overcome through bravery and kindness.

Excited by the possibility of what awaited her, Elara set off on a quest to find the hidden door. Guided by a faint glow that seemed to beckon her, she journeyed through twisting pathways, lush groves, and enchanted glades.

Along the way, she encountered talking animals, wise old trees, and mischievous fairies, each offering clues and riddles that tested her resolve and imagination. With each challenge she faced, Elara grew stronger and more confident, realizing that the true magic lay not just in the world around her, but within herself.

After what felt like days of exploring, she finally found the door—a majestic archway covered in vines and blossoms, with a keyhole that sparkled like the night sky. Heart pounding with excitement, Elara inserted the key. With a gentle turn, the door slowly creaked open, revealing a land more breathtaking than she could have ever imagined.

As she stepped through the doorway, she found herself in a vibrant world filled with colors beyond description, where the sky shimmered in hues of gold and lavender, and the air was filled with the sweet scent of flowers that sang as they swayed in the breeze. Here, she encountered beings of light who welcomed her with open arms.

But soon, she discovered that this realm was in peril. A dark shadow loomed over the land, threatening to steal its magic and joy. Elara knew she couldn’t stand by and do nothing. With the friends she had made along her journey and the courage she had found within herself, she set out to confront the darkness.

Through trials that tested her strength, intellect, and compassion, Elara and her friends gathered the forgotten magic of the realm. They united their powers, confronting the shadow in an epic battle of light and dark. In the end, it was Elara's unwavering belief in hope and friendship that banished the darkness, restoring peace and harmony to the land.

Grateful for her bravery, the beings of light gifted Elara a shimmering pendant that would allow her to return to their world whenever she wished, reminding her that true magic lies in the connections we forge with others and the courage to follow our dreams.

With her heart full of joy, Elara returned to her village, forever changed by her adventure. She would often revisit the magical realm, sharing stories with her friends and inspiring them to embrace their own dreams. And so, the girl who once wandered the woods became a beacon of hope, a reminder that within every heart lies the power to change the world.

And from that day on, the little village thrived, full of laughter, love, and dreams waiting to be explored—each adventure beginning just like hers, with a curious heart and a willingness to believe in the impossible. 

The end.", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None), internal_metrics=[{'cached_prompt_tokens': 0, 'total_accepted_tokens': 0, 'total_batched_tokens': 794, 'total_predicted_tokens': 0, 'total_rejected_tokens': 0, 'total_tokens_in_completion': 795, 'cached_embeddings_bytes': 0, 'cached_embeddings_n': 0, 'uncached_embeddings_bytes': 0, 'uncached_embeddings_n': 0, 'fetched_embeddings_bytes': 0, 'fetched_embeddings_n': 0, 'n_evictions': 0, 'sampling_steps': 767, 'sampling_steps_with_predictions': 0, 'batcher_ttft': 0.20319080352783203, 'batcher_initial_queue_time': 0.12981152534484863}])], created=1737062945, model='gpt-4o-mini-2024-07-18', object='chat.completion', service_tier='default', system_fingerprint='fp_72ed7ab54c', usage=CompletionUsage(completion_tokens=767, prompt_tokens=12, total_tokens=779, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0, cached_tokens_internal=0)))]

#### Example 3: Manual backoff implementation

If you don't want to use third-party libraries, you can implement your own backoff logic.


```python
# imports
import random
import time

# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


completions_with_backoff(model="gpt-4o-mini", messages=[{"role": "user", "content": "Once upon a time,"}])
```

[ChatCompletion(id='chatcmpl-8PAxGvV3GbLpnOoKSvJ00XCUdOglM', choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="in a faraway kingdom, there lived a young princess named Aurora. She was known for her beauty, grace, and kind heart. Aurora's kingdom was filled with lush green meadows, towering mountains, and sparkling rivers. The princess loved spending time exploring the enchanting forests surrounding her castle.

One day, while Aurora was wandering through the woods, she stumbled upon a hidden clearing. At the center stood a majestic oak tree, its branches reaching towards the sky. Aurora approached the tree with curiosity, and as she got closer, she noticed a small door at its base.

Intrigued, she gently pushed open the door and was amazed to find herself in a magical realm. The forest transformed into a breathtaking wonderland, with colorful flowers blooming in every direction and woodland creatures frolicking joyously. Aurora's eyes widened with wonder as she explored this extraordinary world.

As she explored further, Aurora came