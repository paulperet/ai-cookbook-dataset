# Translate a book written in LaTeX from Slovenian into English

With permission of the author, we will demonstrate how to translate the book [Euclidean Plane Geometry](https://sites.google.com/site/projektivna/), written by Milan Mitrović from Slovenian into English, without modifying any of the LaTeX commands.

To achieve this, we will first split the book into chunks, each roughly a page long, then translate each chunk into English, and finally stitch them back together.

## 1. Read in the data


```python
from openai import OpenAI
import tiktoken
client = OpenAI()

# OpenAI tiktoken tokenizer: https://github.com/openai/tiktoken
# we use it to count the number of tokens in the text
tokenizer = tiktoken.get_encoding("o200k_base")

with open("data/geometry_slovenian.tex", "r") as f:
    text = f.read()
```

### 1.1 Count the tokens in each chunk


```python
chunks = text.split('\n\n')
ntokens = []
for chunk in chunks:
    ntokens.append(len(tokenizer.encode(chunk)))
print("Size of the largest chunk: ", max(ntokens))
print("Number of chunks: ", len(chunks))
```

    Size of the largest chunk:  1211
    Number of chunks:  5877


It turns out that a double newline is a good separator in this case, in order not to break the flow of the text. Also no individual chunk is larger than 1211 tokens. The model we will use is gpt-4o, which has a limit of 16,384 tokens, so we don't need to worry about breaking the chunks down further.

We will group the shorter chunks into chunks of around 15000 tokens, to increase the coherence of the text, and decrease the frequency of breaks within the text.


```python
def group_chunks(chunks, ntokens, max_len=15000, hard_max_len=16000):
    """
    Group very short chunks, to form approximately page long chunks.
    """
    batches = []
    cur_batch = ""
    cur_tokens = 0
    
    # iterate over chunks, and group the short ones together
    for chunk, ntoken in zip(chunks, ntokens):
        # discard chunks that exceed hard max length
        if ntoken > hard_max_len:
            print(f"Warning: Chunk discarded for being too long ({ntoken} tokens > {hard_max_len} token limit). Preview: '{chunk[:50]}...'")
            continue

        # if room in current batch, add new chunk
        if cur_tokens + 1 + ntoken <= max_len:
            cur_batch += "\n\n" + chunk
            cur_tokens += 1 + ntoken  # adds 1 token for the two newlines
        # otherwise, record the batch and start a new one
        else:
            batches.append(cur_batch)
            cur_batch = chunk
            cur_tokens = ntoken
            
    if cur_batch:  # add the last batch if it's not empty
        batches.append(cur_batch)
        
    return batches


chunks = group_chunks(chunks, ntokens)
len(chunks)
```




    39



Notice that adding a sample untranslated and translated first command, where only the content of the chapter name needs to be translated, helps to get more consistent results.

The format of the prompt sent to the model consists of:
1. A high level instruction to translate only the text, but not commands into the desired language
2. A sample untranslated command, where only the content of the chapter name needs to be translated
3. The chunk of text to be translated
4. The translated sample command from 2, which shows the model the beginning of the translation process

The expected output is the translated chunk of text.


```python
def translate_chunk(chunk, model='gpt-4o',
                    dest_language='English',
                    sample_translation=(
                    r"\poglavje{Osnove Geometrije} \label{osn9Geom}",
                    r"\chapter{The basics of Geometry} \label{osn9Geom}")):
    prompt = f'''Translate only the text from the following LaTeX document into {dest_language}. Leave all LaTeX commands unchanged
    
"""
{sample_translation[0]}
{chunk}"""

{sample_translation[1]}
'''
    response = client.chat.completions.create(
        messages=[{"role": "user", "content":prompt}],
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=15000,
    )
    result = response.choices[0].message.content.strip()
    result = result.replace('"""', '') # remove the double quotes, as we used them to surround the text
    return result
print(translate_chunk(chunks[2], model='gpt-4o', dest_language='English'))
```

    Certainly! Here's the translation of the text from the LaTeX document into English, with all LaTeX commands unchanged:
    
    ---
    
    \chapter{The basics of Geometry} \label{osn9Geom}
    Let us mention that the group structure also requires the property of associativity, i.e., $\mathcal{I}_1\circ (\mathcal{I}_2\circ \mathcal{I}_3)= (\mathcal{I}_1\circ \mathcal{I}_2)\circ \mathcal{I}_3$ (for arbitrary isometries $\mathcal{I}_1$, $\mathcal{I}_2$, and $\mathcal{I}_3$), which is automatically fulfilled in the operation of function composition. Let us also mention that the \concept{identity} \index{identity} $\mathcal{E}$ from the previous axiom is a mapping for which $\mathcal{E}(A)=A$ for every point of the plane. The mapping $\mathcal{I}^{-1}$ is the \concept{inverse mapping} for the isometry $\mathcal{I}$ if $\mathcal{I}^{-1}\circ \mathcal{I} =\mathcal{I}\circ\mathcal{I}^{-1}=\mathcal{E}$. According to the previous axiom, the identity and inverse mapping of each isometry are also isometries.
    
    Let us prove the first consequences of the axioms of congruence. First, we will consider the following properties of isometries.
    
    \begin{theorem} \label{izrekIzoB} Isometry maps a line to a line, a line segment to a line segment, a ray to a ray, a half-plane to a half-plane, an angle to an angle, and an $n$-gon to an $n$-gon.
    \end{theorem}
    
    \textbf{\textit{Proof.}}
    According to axiom \ref{aksIII1}, isometries preserve the relation $\mathcal{B}$. Therefore, all points of the line segment $AB$ under the isometry $I$ are mapped to points lying on the line segment $A'B'$, where $A'=\mathcal{I}(A)$ and $B'=\mathcal{I}(B)$. Since the inverse mapping $\mathcal{I}^{-1}$ is also an isometry (axiom \ref{aksIII4}), every point of the line segment $A'B'$ is the image of some point lying on the line segment $AB$. Thus, the line segment $AB$ is mapped to the line segment $A'B'$ by the isometry $\mathcal{I}$.
    
    The remaining figures from the theorem are also defined using the relation $\mathcal{B}$, so the proof is similar to that for the line segment.
    \qed
    
    From the proof of the previous theorem, it follows that the endpoints of the line segment $AB$ are mapped to the endpoints of the image $A'B'$ by the isometry. Similarly, the origin of a ray is mapped to the origin of the ray, the edge of a half-plane to the edge of a half-plane, the vertex of an angle to the vertex of an angle, and the vertex of a polygon to the vertex of a polygon.
    
    Isometries are defined as bijective mappings that preserve the congruence of pairs of points. Is it also true that for congruent pairs of points, there exists an isometry that maps the first pair to the second? Let us provide the answer with the following theorem.
    
    \begin{theorem} \label{izrekAB} If $(A,B)\cong (A',B')$, then there is an isometry $\mathcal{I}$, which maps the points $A$ and $B$ to the points $A'$ and $B'$, i.e.:
    $$\mathcal{I}: A, B\mapsto A',B'.$$
    \end{theorem}
    
    \begin{figure}[!htb]
    \centering
    \input{sl.aks.2.3.7.pic}
    \caption{} \label{sl.aks.2.3.7.pic}
    \end{figure}
    
    \textbf{\textit{Proof.}}
    Let $C$ be a point that does not lie on the line $AB$, and $C'$ a point that does not lie on the line $A'B'$ (Figure \ref{sl.aks.2.3.7.pic}). According to axiom \ref{aksIII2}, there exists a unique isometry $\mathcal{I}$ that maps the point $A$ to the point $A'$, the ray $AB$ to the ray $A'B'$, and the half-plane $ABC$ to the half-plane $A'B'C'$. Since by assumption $(A,B)\cong (A',B')$ from the same axiom \ref{aksIII2}, it follows that $\mathcal{I}(B)=B'$.
    \qed
    
    The proof of the following theorem is similar, which will later be presented in a different form as the first theorem on the congruence of triangles.
    
    \begin{theorem} \label{IizrekABC} Let $(A,B,C)$ and $(A',B',C')$ be triplets of non-collinear points such that $$(A,B,C)\cong (A',B',C'),$$ then there is a single isometry $\mathcal{I}$, that maps the points $A$, $B$, and $C$ into the points $A'$, $B'$, and $C'$, i.e.:
    $$\mathcal{I}: A, B,C\mapsto A',B',C'.$$
    \end{theorem}
    
    \begin{figure}[!htb]
    \centering
    \input{sl.aks.2.3.5.pic}
    \caption{} \label{sl.aks.2.3.5.pic}
    \end{figure}
    
    \textbf{\textit{Proof.}}
    According to axiom \ref{aksIII2}, there exists a unique isometry $\mathcal{I}$ that maps the point $A$ to the point $A'$, the ray $AB$ to the ray $A'B'$, and the half-plane $ABC$ to the half-plane $A'B'C'$ (Figure \ref{sl.aks.2.3.5.pic}). Since by assumption $(A,B,C)\cong (A',B',C')$ from the same axiom \ref{aksIII2}, it follows that $\mathcal{I}(B)=B'$ and $\mathcal{I}(C)=C'$.
    
    It is necessary to prove that $\mathcal{I}$ is the only such isometry. Suppose there exists such an isometry $\mathcal{\widehat{I}}$ that satisfies $\mathcal{\widehat{I}}: A, B,C\mapsto A',B',C'$. According to theorem \ref{izrekIzoB}, the isometry $\mathcal{\widehat{I}}$ also maps the ray $AB$ to the ray $A'B'$ and the half-plane $ABC$ to the half-plane $A'B'C'$. From axiom \ref{aksIII2}, it follows that $\mathcal{\widehat{I}}=\mathcal{I}$.
    \qed
    
    A direct consequence is the following theorem.
    
    \begin{theorem} \label{IizrekABCident} Let $A$, $B$, and $C$ be three non-collinear points, then the identity map $\mathcal{E}$ is the only isometry that maps points $A$, $B$, and $C$ to the same points $A$, $B$, and $C$.
    \end{theorem}
    
    \begin{figure}[!htb]
    \centering
    \input{sl.aks.2.3.5a.pic}
    \caption{} \label{sl.aks.2.3.5a.pic}
    \end{figure}
    
    \textbf{\textit{Proof.}} (Figure \ref{sl.aks.2.3.5a.pic})
    
    First, the identical mapping $\mathcal{E}$, which maps the points $A$, $B$, and $C$ to the points $A$, $B$, and $C$, is an isometry according to axiom \ref{aksIII4}. From the previous theorem \ref{IizrekABC}, it follows that there is only one such isometry.
    \qed
    
    For the point $A$, we say that it is a \index{point!fixed} \concept{fixed point} (or \index{point!immovable} \concept{immovable point}) of the isometry $\mathcal{I}$ if $\mathcal{I}(A)=A$. The previous theorem tells us that the only isometries that have three non-collinear fixed points are identities.
    
    We will discuss isometries in more detail in chapter \ref{pogIZO}, but for now, we will use them primarily to help introduce the congruence of figures. Two figures $\Phi$ and $\Phi'$ are \index{figures!congruent}\concept{congruent} (denoted $\Phi\cong \Phi'$) if there exists an isometry $I$ that maps the figure $\Phi$ to the figure $\Phi'$.
    
    A direct consequence of axiom \ref{aksIII4} is the following theorem.
    
    \begin{theorem}
    Congruence of figures is an equivalence relation. \label{sklRelEkv}
    \end{theorem}
    
    \textbf{\textit{Proof.}}
    
    \textit{Reflexivity.} For every figure $\Phi$, it holds that $\Phi \cong \Phi$, because the identical mapping $\mathcal{E}$ is an isometry (axiom \ref{aksIII4}) and $\mathcal{E}:\Phi\rightarrow\Phi$.
    
    \textit{Symmetry.} From $\Phi \cong \Phi_1$, it follows that there exists an isometry $\mathcal{I}$ that maps the figure $\Phi$ to the figure $\Phi_1$. The inverse mapping $\mathcal{I}^{-1}$, which is an isometry according to axiom \ref{aksIII4}, maps the figure $\Phi_1$ to the figure $\Phi$, so $\Phi_1 \cong \Phi$.
    
    \textit{Transitivity.} From $\Phi \cong \Phi_1$ and $\Phi_1 \cong \Phi_2$, it follows that there exist such isometries $\mathcal{I}$ and $\mathcal{I}'$ that satisfy $\mathcal{I}:\Phi\rightarrow\Phi_1$ and $\mathcal{I}':\Phi_1\rightarrow\Phi_2$. Then the composition $\mathcal{I}'\circ\mathcal{I}$, which is an isometry according to axiom \ref{aksIII4}, maps the figure $\Phi$ to the figure $\Phi_2$, so $\Phi \cong \Phi_2$.
    \qed
    
    The concept of congruence of figures also applies to line segments. Intuitively, we have associated the congruence of line segments with the congruence of pairs of points. Now we will prove the equivalence of both relations.
    
    \begin{theorem} \label{izrek(A,B)} $AB \cong A'B' \Leftrightarrow (A,B)\cong (A',B')$
    \end{theorem}
    
    \textbf{\textit{Proof.}}
    
    ($\Rightarrow$) If $(A,B)\cong (A',B')$, according to theorem \ref{izrekAB}, there exists an isometry $\mathcal{I}$ that maps the points $A$ and $B$ to the points $A'$ and $B'$. From theorem \ref{izrekIzoB}, it follows that the isometry $\mathcal{I}$ maps the line segment $AB$ to the line segment $A'B'$, i.e., $AB \cong A'B'$.
    
    ($\Leftarrow$) If $AB \cong A'B'$, there exists an isometry $\mathcal{I}$ that maps the line segment $AB$ to the line segment $A'B'$. According to the consequence of theorem \ref{izrekIzoB}, the endpoint of the line segment is mapped to the endpoint of the line segment. This means that either $\mathcal{I}:A,B\mapsto A',B'$ or $\mathcal{I}:A,B\mapsto B',A'$. From the first relation, it follows that $(A,B)\cong (A',B')$, and from the second, $(A,B)\cong (B',A')$. However, even from the second case, we get $(A,B)\cong (A',B')$, which is a consequence of axioms \ref{aksIII3} and \ref{aksIII4}.
    \qed
    
    Due to the previous theorem, we will always write $AB\cong A'B'$ instead of the relation $(A,B)\cong (A',B')$ in the continuation.
    
    \begin{theorem} \label{ABnaPoltrakCX}
    For each line segment $AB$ and each ray $CX$, there is exactly one point $D$ on the ray $CX$ that $AB\cong CD$ holds.
    \end{theorem}
    
    \begin{figure}[!htb]
    \centering
    \input{sl.aks.2.3.5b.pic}
    \caption{} \label{sl.aks.2.3.5b.pic}
    \end{figure}
    
    \textbf{\textit{Proof.}} Let $P$ be a point that does not lie on the line $AB$, and $Q$ a point that does not lie on the line $CX$ (Figure \ref{sl.aks.2.3.5b.pic}). According to axiom \ref{aksIII2}, there exists a unique isometry $\mathcal{I}$ that maps the point $A$ to the point $C$, the ray $AB$ to the ray $CX$, and the half-plane $ABP$ to the half-plane $CXQ$. Let $D=\mathcal{I}(C)$, then $AB \cong CD$.
    
    Assume that there is another point $\widehat{D}$ on the ray $CX$ for which $AB \cong C\widehat{D}$. Since