# Gemini API: Name of your guide

<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Template.ipynb"></a>

_[remove me] Be sure to update the Colab link!_

[If you're adding a new example, use this badge to promote yourself (yes sorry you'll have to write your name a lot of times)]
<!-- Community Contributor Badge -->
This notebook was contributed by Giom (use either your GitHub handle or your real name, your choice).
Add a link to you blog, or linkedIn, or something else - See Giom other notebooks here.
Have a cool Gemini example? Feel free to share it too!

[Depending on the cases you might also want to add a badge like that as a disclaimer]

<!-- Princing warning Badge -->
⚠️
Image generation is a paid-only feature and won't work if you are on the free tier. Check the pricing page for more details.

<!-- Notice Badge -->
🪧
Image-out is a preview feature. It is free to use for now with quota limitations, but is subject to change.

[Include a paragraph or two here explaining what this example demonstrates, who should be interested in it, and what you need to know before you get started.]

## Setup

### Install SDK


```
%pip install -U -q "google-genai>=1.0.0"  # Install the Python SDK

# Always set at least 1.0.0 as the minimal version as there were breaking
# changes through the previous versions
# Of course, if your notebook uses a new feature and needs a more recent
# version, set the right minimum version to indicate when the feature was
# introduced.
# Always test your notebook with that fixed version (eg. '==1.0.0') to make.
# sure it's really the minimum version.

```

### Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication ](../quickstarts/Authentication.ipynb) quickstart for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Now select the model you want to use in this guide, either by selecting one in the list or writing it down. Keep in mind that some models, like the 2.5 ones are thinking models and thus take slightly more time to respond (cf. [thinking notebook](./Get_started_thinking.ipynb) for more details and in particular learn how to switch the thiking off).


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

# Ideally order the model by "cabability" ie. generation then within generation
# 8b/flash-lite then flash then pro
```

## [Write your guide]

[Add as many high level sections as needed to step through your guide. Try to introduce new concepts incrementally, including explanatory text at every step. Remember that notebooks need to be executable from start to finish using `Runtime -> Run all` in Colab.]

## Next Steps
### Useful API references:

[Always end with links to the related doumentation]

### Related examples

[If any, add links to the related examples]

### Continue your discovery of the Gemini API

[Finally, link some other quickstarts that are either related or require the same level of expertise]


# Resources (don't forget to delete everything starting from here)

* Follow the [Google developer documentation style guide](https://developers.google.com/style/highlights)
* The [TensorFlow documentation style guide](https://www.tensorflow.org/community/contribute/docs_style) has useful guidance for notebooks.
* Read the [Cookbook contributor guide](https://github.com/google-gemini/cookbook/blob/main/CONTRIBUTING.md) and the [Cookbook Examples contributor guide](https://github.com/google-gemini/cookbook/blob/main/examples/CONTRIBUTING.md).

## Notebook style (also check the [Contributing guide](Contributing.mg))

* Include the collapsed license at the top (uses the Colab "Form" mode to hide the cells).
* Save the notebook with the table of contents open.
* Use one `H1` header for the title.
* Include the button-bar immediately after the `H1`.
* Include an overview section before any code.
* Keep your installs and imports close to the code that first uses them. If they are used throughout (such as the SDK), they can go at the start of the guide.
* Keep code and text cells as brief as possible.
* Break text cells at headings
* Break code cells between "building" and "running", and between "printing one result" and "printing another result".
* Necessary but uninteresting code should be hidden in a toggleable code cell by putting `# @title` as the first line.
* You can optionally add a byline for content in `examples/` that you wrote, including one link to your GitHub, social, or site of your choice.

### Code style

* Notebooks are for people. Write code optimized for clarity.
* Keep examples quick and concise.
* Use the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html), where applicable. Code formatted by [`pyink`](https://github.com/google/pyink) will always be accepted.
* In particular, defining functions and variables takes extra spaces around the `=` sign, while function parameters don't:
```python
    var = value
    function(
      parameter=value
    )
```
* If you define a function, run it and show us what it does before using it in another function.
* Demonstrate small parts before combining them into something more complex.
* Keep the code as simple as possible, only use extra parameters like temperature when needed, and in that case, explain why
* To ensure notebook text remains accurate, present model metadata by executing code.
  * For example, instead of saying "1M token context" in the text, display the output of `client.models.get(model='...').input_token_limit`.


### Text

* Use an imperative style. "Run a prompt using the API."
* Use sentence case in titles/headings.
* Use short titles/headings: "Download the data", "Call the API", "Process the results".
* Use the [Google developer documentation style guide](https://developers.google.com/style/highlights).
* Use [second person](https://developers.google.com/style/person): "you" rather than "we".
* When using links between notebooks, use relative ones as they'll work better in IDEs and Colab. Use absolute ones to link to folders or markdown files.


## GitHub workflow

* Be consistent about how you save your notebooks, otherwise the JSON diffs are messy. [`nbfmt` and `nblint`](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/tools/README.md) help here.
* This notebook has the "Omit code cell output when saving this notebook" option set. GitHub refuses to diff notebooks with large diffs (inline images).
* [ReviewNB.com](http://reviewnb.com) can help with diffs. This is linked in a comment on a notebook pull request.
* Use the [Open in Colab](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) extension to open a GitHub notebook in Colab.
* The easiest way to edit a notebook in GitHub is to open it with Colab from the branch you want to edit. Then use File --> Save a copy in GitHub, which will save it back to the branch you opened it from.
* For PRs it's helpful to post a direct Colab link to the PR head: https://colab.research.google.com/github/{USER}/{REPO}/blob/{BRANCH}/{PATH}.ipynb