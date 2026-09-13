"""Prompt construction shared by the finetuning and inference notebooks.

Both import from here so the format cannot drift between training and
inference. A mismatch degrades results silently and is hard to spot.
"""

LEVELS = ["A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+"]
LEVELS_STR = ", ".join(LEVELS)

SYSTEM_PROMPT = """You are an expert in assessing written English produced by learners of English as a foreign language.
You will assign a CEFR level to learner writing samples using the following official CEFR descriptors for Overall Written Production.

CEFR Overall Written Production Descriptors (Council of Europe, 2020):

C2: Can produce clear, smoothly flowing, complex texts in an appropriate and effective style 
and a logical structure which helps the reader identify significant points.

C1: Can produce clear, well-structured texts of complex subjects, underlining the relevant 
salient issues, expanding and supporting points of view at some length with subsidiary points, 
reasons and relevant examples, and rounding off with an appropriate conclusion.
Can employ the structure and conventions of a variety of genres, varying the tone, style and 
register according to addressee, text type and theme.

B2: Can produce clear, detailed texts on a variety of subjects related to their field of 
interest, synthesising and evaluating information and arguments from a number of sources.

B1: Can produce straightforward connected texts on a range of familiar subjects within their 
field of interest, by linking a series of shorter discrete elements into a linear sequence.

A2: Can produce a series of simple phrases and sentences linked with simple connectors 
like "and", "but" and "because".

A1: Can give information about matters of personal relevance (e.g. likes and dislikes, family, 
pets) using simple words and basic expressions.
Can produce simple isolated phrases and sentences.

Plus levels represent a very strong competence at a level that does not yet reach the minimum 
standard for the next criterion level. Generally, features of the level above are starting to appear.

The levels you should assign are: A2, A2+, B1, B1+, B2, B2+, C1, C1+."""

CLASSIFICATION_REQUEST = (
    f"Assign a CEFR level to the following learner writing. "
    f"The possible levels are: {LEVELS_STR}.\n\n"
    "Respond with the level only.\n\n"
    "Text to classify:\n\n"
)


def build_messages(text, label=None):
    """Chat messages for one example. Pass `label` for training, omit for inference."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CLASSIFICATION_REQUEST + str(text)},
    ]
    if label is not None:
        msgs.append({"role": "assistant", "content": label})
    return msgs
