### Abstract
We present the Orthogonal Transformer, a transformer model in which residual additions are replaced with orthogonal transformations. This allows for perfect norm preservation of activation vectors between layers at any training step and nearly constant layer-wise gradient norms at initialization. We view the output of each residual block (attention or MLP) and the block’s input activation as two components of a bivector that defines a rotation in the two-dimensional plane they span. This perspective naturally recovers and unifies recent modifications such as residual stream rescaling and orthogonal residual updates. We test our approach on a small language model (~10M parameters) with character-level tokenization and a small dataset (~1M tokens), on which it achieves results comparable to those of a standard transformer. Larger-scale experiments are forthcoming.

### Introduction

The neural network architecture can significantly affect its signal propagation properties, which in turn shape its training dynamics [[Schoenholz et al., 2017]; [Poole et al., 2016]; [Cowsik et al., 2024]]. For a given architecture, each point in the hyperparameter space may be characterized by a quantity called the critical length, which limits the maximum depth of effective signal propagation through the network at initialization. It has been proposed that the best trainability is achieved at the edge of chaos (criticality), where the critical length diverges. Away from criticality, perturbations in the forward pass, as well as gradients in the backward pass, grow or decay exponentially with depth. At criticality, both instead vary according to a power law with some critical exponent [[Doshi et al., 2023]]. Residual connections and pre-layer normalization make a larger region of hyperparameter space critical [[Doshi et al., 2023]], which in practice enables good trainability with less extensive hyperparameter tuning.

Gradient propagation is closely related to forward signal propagation. Indeed, the partial Jacobian matrix between layers $l_0$ and $l$, with $l>l_0$, both locally approximates the chain of layer-wise transformations and directly determines how gradients are transformed when they are propagated from layer $l$ to layer $l_0$ [[Schoenholz et al., 2017]; [Doshi et al., 2023]]. In particular, preserving activation norms in the forward pass is related to maintaining stable gradient norms during backpropagation [[Doshi et al., 2023]; [Kedia et al., 2024]]. A large body of work studies how to build very deep transformers that rely on either post-normalization or residual stream scaling to provide more control over the gradients, at least at initialization [[Kedia et al., 2024]; [Wang et al., 2022]]. Recent works such as [[Kim et al., 2025]; [Oh et al., 2025]] empirically show that normalizing residual block outputs, or projecting out their radial component aligned with the activations, can improve training stability.

We present a new orthogonal residual update rule for transformers that exactly preserves activation norms in the forward pass at any training step and yields stable layer-wise gradients at initialization. We interpret the input activation to each residual block and its output (attention or MLP) as forming a bivector that generates a rotation in the two-dimensional plane they span. From this viewpoint, techniques like residual stream rescaling and orthogonal residual updates emerge naturally and can be viewed within a single unified framework. A similar geometric construction has recently been applied to positional encodings in transformers [[Zhang et al., 2025]].

### Orthogonal Transformer

#### Background

Let $x_s^{(l)} \in \mathbb{R}^d$ denote the token representation at residual layer $l$ and sequence position $s$. A transformer with $L$ transformer blocks and pre-layer RMS normalization can be viewed as a sequence of $2L$ alternating updates of the form

$$
\begin{equation}
\begin{aligned}
x^{(l+1)}_s &= x^{(l)}_s + u^{(l)}_s\left(\{\tilde x^{(l)}_{s'}\}_{s'=1}^S\right), && \text{for even } l,\\
x^{(l+1)}_s &= x^{(l)}_s + u^{(l)}\left(\tilde x^{(l)}_s\right), && \text{for odd } l,
\end{aligned}
\end{equation}
$$

where $\tilde x_s^{(l)} = \mathrm{RMSNorm}(x_s^{(l)})$ and $S$ is the sequence length. For even $l$, $u_s^{(l)}$ is an attention block whose input is the entire sequence of normalized token representations and whose functional form depends on the position $s$. For odd $l$, $u^{(l)}$ is an MLP block that acts on the normalized token representation at position $s$ with a transformation that does not depend on the positional index. Finally, $x_s^{(0)}$ denotes the initial token embedding at position $s$. Geometrically, RMSNorm maps token representations onto the $(d-1)$-dimensional sphere of radius $\sqrt d$, $S^{d-1}_{\sqrt{d}}$, followed by a learnable diagonal linear map. In what follows, we will not distinguish between MLP and attention transformations and will write both simply as

$$
\begin{equation}
x^{(l+1)} = x^{(l)} + u^{(l)},
\end{equation}
$$

omitting the positional index $s$ as well.

The vector $u^{(l)} \in \mathbb{R}^d$ can be decomposed into radial and tangential components, $u^{(l)} = u_{\parallel}^{(l)} + u_{\perp}^{(l)}$, with the radial component aligned with $x^{(l)}$ and the tangential component orthogonal to $x^{(l)}$ in the sense of the standard inner product on $\mathbb{R}^d$. Before mapping representation vectors to logits, the output of the final transformer block $x^{(2L)}$ is mapped to the sphere $S^{d-1}_{\sqrt{d}}$ for the last time. Thus, radial motion has no direct effect on the final logits produced by the model, as it is completely projected out by the final RMSNorm.

On the other hand, radial motion does affect gradient backpropagation. First, the final RMSNorm rescales the gradient passing through it with a coefficient proportional to $1 / \lVert x^{(2L)} \rVert$, the inverse norm of the pre-RMSNorm activation vector, which can be seen from its Jacobian:

$$
\begin{equation}
\frac{\partial \mathrm{RMSNorm}^i}{\partial x^j}
= \frac{\sqrt{d}}{\lVert x\rVert}\left(\delta_{ij}-\frac{x_i x_j}{\lVert x\rVert^2}\right).
\end{equation}
$$

Second, the law governing layer-wise activation norm dynamics also determines the law governing layer-wise gradient norm dynamics. At initialization, layer-wise activation norms approximately follow the square-root law $\lVert x^{(l)} \rVert \sim \sqrt{l}$, which results in the corresponding gradient norms following the inverse square-root law $\lVert g^{(l)} \rVert \sim 1 / \sqrt{l}$. The first statement follows directly from Eq. (2), applied recursively to the initial token embedding, under the assumption that the velocities $u^{(l)}$ are approximately orthogonal to the representation vectors $x^{(l)}$, which is justified by the random Gaussian initialization of the weight matrices in the linear maps of the MLP and attention blocks. For a proof of the second statement, we refer to [[Doshi et al., 2023]; [Kedia et al., 2024]].

[[Oh et al., 2025]] empirically demonstrate that the update rule $x^{(l+1)} = x^{(l)} + u_\perp^{(l)}$, which projects out the radial component of the residual block output, can improve generalization accuracy and training stability. [[Kedia et al., 2024]] achieve stable training of a 1000-layer transformer via residual rescaling, using the update rule $x^{(l+1)} = \lambda x^{(l)} + \beta u^{(l)}$, where $\lambda^2 + \beta^2 = 1$ and $x^{(l)}$ and $u^{(l)}$ are assumed to be orthogonal and to have approximately the same norm, which can be achieved through proper initialization. In the next section, we show that these two approaches can be unified within a single framework.

#### Orthogonal Transformer update rule

Let us first assume, for conceptual clarity, that we want to ensure $\lVert x^{(l)} \rVert = 1$ at each layer. In our practical implementation, considered in the next section, we instead choose to keep $\lVert x^{(l)} \rVert = \sqrt{d}$, as this is more natural given the standard definition of RMSNorm.

Instead of directly adding the residual block output $u^{(l)}$ to the residual stream $x^{(l)}$, we propose viewing the two-dimensional plane they span as a rotation plane, with the rotation angle defined by the magnitude of $\lVert u_{\perp}^{(l)} \rVert$. Formally, for a pair $(x, u)$ with $\lVert x \rVert = 1$, we define the bivector $b = x \wedge u$, which naturally maps to an element $B(x, u)$ of the Lie algebra $\mathfrak{so}(d)$ of the Lie group $SO(d)$: $B(x, u) = x u^T - u x^T$.
The expression on the right-hand side does not depend on the radial component of $u$, so we can equivalently write $B(x, u) = x u_{\perp}^T - u_{\perp} x^T$.
Defining the unit vector $\hat u_\perp = u_\perp / {\lVert u_\perp \rVert}$ and omitting the arguments of $B$, we obtain

$$
\begin{equation}
B = \theta \hat{B},
\end{equation}
$$

where $\theta = \lVert u_\perp \rVert$ and $\hat B = x \hat u_{\perp}^T - \hat u_{\perp} x^T$ is constructed from the pair of orthogonal unit vectors $(x, \hat u_\perp)$. Exponentiating $B$ yields a rotation in $SO(d)$ in the two-dimensional plane spanned by $x$ and $u$:

$$
\begin{equation}
\exp(B) = I + \hat{B} \sin\theta + \hat{B}^2 (1-\cos\theta).
\end{equation}
$$

We now define the transformation in the $l$-th transformer layer as

$$
\begin{equation}
x^{(l+1)} = \exp\left(B(x^{(l)},u^{(l)})\right)x^{(l)}.
\end{equation}
$$

Using the explicit form of the exponential map, the expression for $x^{(l+1)}$ simplifies to

$$
\begin{equation}
x^{(l+1)} = x^{(l)}\, \cos \theta + \hat{u}^{(l)}_{\perp}\, \sin \theta,
\end{equation}
$$

with $\theta = \lVert u_\perp^{(l)} \rVert$. By construction, this map exactly preserves the unit norm of the representation vectors, given that $\lVert x^{(0)} \rVert = 1$. We emphasize that projecting out the radial component of the MLP/attention output and residual rescaling arise naturally in this approach and are unified by it. The novelty here is that, instead of using constant rescaling coefficients treated as hyperparameters requiring tuning, the coefficients are dynamically determined by the magnitude of the tangential component of $u^{(l)}$.

Note that Eq. (7) can be written in terms of the non-normalized generator $u^{(l)}_{\perp}$ as follows:

$$
\begin{equation}
x^{(l+1)} = x^{(l)}\, \cos \theta + u^{(l)}_\perp\, \frac{\sin \theta}{\theta}.
\end{equation}
$$

In the limit of small $\theta$, this update rule reproduces the residual addition formula with the radial component of $u^{(l)}$ projected out: $x^{(l+1)} = x^{(l)} + u_\perp^{(l)}$.

#### Implementation

In practice, we choose to keep $\lVert x^{(l)} \rVert = \sqrt{d}$, so we redefine $b = \frac{x}{\sqrt{d}} \wedge \frac{\hat u_\perp}{\sqrt{d}}$.
Thus, $B = \theta \hat{B}$, where $\theta = \lVert u_\perp \rVert / \sqrt{d}$ and $\hat B = x \hat u_{\perp}^T - \hat u_{\perp} x^T$ is constructed from the pair of orthogonal unit vectors $(\hat{x}, \hat u_\perp)$ as above, with $\hat{x} = x / \sqrt{d}$. Equations (5) and (8) retain their form with the redefined $\theta$. To avoid division by zero in Eq. (8), we introduce a threshold $\epsilon$ such that, when $\theta < \epsilon$, we replace the update rule with the residual addition formula $x^{(l+1)} = x^{(l)} + u_\perp^{(l)}$. 

### Experiments

Both for the baseline and for the Orthogonal Transformer, we train 16-layer transformers with embedding dimension $d = 256$ and number of attention heads $n_{\text{heads}} = 4$, sweeping over weight initializations. We train them on a language modeling task on the Tiny Shakespeare dataset (1M tokens) with the Adam optimizer ($\text{lr} = 0.004$, $\beta_1=0.9$, $\beta_2 = 0.99$, no weight decay). We initialize the value matrix and output projection matrix in the attention blocks from a Gaussian distribution with variance $\sigma_w^2 / d$. The two matrices in the MLP, $W_1^{d \to 4d}$ and $W_2^{4d \to d}$, are initialized with variances $\sigma_w^2 / d$ and $2\sigma_w^2 / 4d$, respectively (the factor of $2$ in the latter compensates for the $\text{ReLU}$ variance-shrinking factor of $1/2$). The query and key matrices are initialized with variances $\sigma_{qk}^2 / d$, with $\sigma_{qk}$ independent of $\sigma_w$. This choice of initialization allows us to control the magnitude of attention/MLP outputs $u^{(l)}$ at all layers simultaneously, without affecting the attention mechanism itself, i.e., the computation of the attention matrices. Finally, the token embeddings are initialized with component-wise variance $1$.

The baseline transformer uses pre-layer RMS normalization, with no removal of the residual block’s radial component and no residual scaling. In the Orthogonal Transformer, we introduce a single RMSNorm layer with non-learnable weights after the token embeddings and remove all other RMSNorm layers, including the final one, since the residual stream remains normalized under the update rule.

Fig. [] shows that both models achieve comparable validation loss on the held-out subset of the Tiny Shakespeare dataset, with the Orthogonal Transformer showing slightly lower validation loss. Fig. [] compares layer-wise gradient norms at initialization between the two approaches, showing that the Orthogonal Transformer has approximately constant layer-wise gradient norms.

Despite the stability of gradients at initialization in the Orthogonal Transformer, we find that larger values of $\sigma_w$ at initialization may lead to exponentially vanishing gradients at later training steps (Fig. []). The underlying mechanism of this instability remains unclear, and we leave its study for future work. This finding emphasizes that perfect norm preservation of activations and perfect gradient propagation at initialization do not guarantee that this behavior will persist during training (see also [[Pennington et al., 2017]; [Bansal et al., 2018]; [Bachlechner et al., 2020]]), which calls for further studies aimed at a better understanding of training dynamics of deep neural networks. It would also be interesting to compare our approach directly to existing methods that use residual block radial component removal or residual scaling.

### Conclusion

We presented the Orthogonal Transformer, which replaces residual additions with data-dependent orthogonal rotations defined by the plane spanned by the residual stream $x^{(l)}$ and the residual block output $u^{(l)}$. This update rule exactly preserves activation norms at every layer and training step, while naturally recovering and unifying ideas like radial-component removal and residual rescaling, with coefficients determined dynamically by the tangential residual magnitude rather than tuned hyperparameters. 

On a small language-modeling setup (16 layers, 
∼10M parameters, Tiny Shakespeare 
∼1M tokens), the Orthogonal Transformer achieved comparable validation loss to a standard transformer and showed approximately constant layer-wise gradient norms at initialization. However, we found that larger initialization scales can still yield vanishing gradients later in training, indicating that norm preservation and favorable initialization-time propagation do not alone ensure stable optimization throughout training. Future work will investigate this instability and validate the approach at larger scales.

### References
See [REFERENCES.md](./REFERENCES.md)

[Schoenholz et al., 2017]: ./REFERENCES.md#ref-schoenholz2017deepinformationpropagation  
[Poole et al., 2016]: ./REFERENCES.md#ref-poole2016exponentialexpressivitydeepneural  
[Cowsik et al., 2024]: ./REFERENCES.md#ref-cowsik2024geometricdynamicssignalpropagation  
[Doshi et al., 2023]: ./REFERENCES.md#ref-doshi2023criticalinitializationwidedeep  
[Kedia et al., 2024]: ./REFERENCES.md#ref-kedia2024transformersstableendtoendsignal  
[Wang et al., 2022]: ./REFERENCES.md#ref-wang2022deepnetscalingtransformers1000  
[Kim et al., 2025]: ./REFERENCES.md#ref-kim2025perilnrevisitingnormalizationlayer  
[Oh et al., 2025]: ./REFERENCES.md#ref-oh2025revisitingresidualconnectionsorthogonal  
[Zhang et al., 2025]: ./REFERENCES.md#ref-zhang2025grouprepresentationalpositionencoding  
[Xiong et al., 2020]: ./REFERENCES.md#ref-xiong2020layernormalizationtransformerarchitecture  
[Zhang and Sennrich, 2019]: ./REFERENCES.md#ref-zhang2019rootmeansquarelayer
[Bansal et al., 2018]: ./REFERENCES.md#ref-bansal2018gainorthogonalityregularizationstraining
[Pennington et al., 2017]: ./REFERENCES.md#ref-pennington2017resurrectingsigmoiddeeplearning
[Bachlechner et al., 2020]: ./REFERENCES.md#ref-bachlechner2020rezeroneedfastconvergence

