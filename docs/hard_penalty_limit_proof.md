# The hard-penalty limit of the penalized Saito functional

**Theorem.** Fix a line arrangement $\mathcal A$ of $n$ distinct lines, a
target degree pair with $d_1\ge 0$, $d_2\ge 0$, $d_1+d_2=n-1$ — treated as
an **unordered** pair, normalized throughout by the convention $d_1\le d_2$
— and $0<\beta<1$.

*Degree-zero boundary.* $S_{-1}=\{0\}$, so $\mu_{\alpha,0}\colon\{0\}\to
S_0$ is the zero map, $\alpha S_{-1}=\{0\}$, and $\Pi_{\alpha,0}=0$: the
degree-0 residual is $\rho_{\alpha,0}(u)=u(\alpha)$ itself. All statements
below include $d_1=0$.
With the Bombieri–Weyl normalizations of `penalized_saito.py`,

$$S_{\lambda,\beta}(\mathcal A; d_1,d_2)\;=\;1-\sup_{\|u\|=\|v\|=1}
\Gamma_{\lambda,\beta}(u,v),$$

the following hold for the **ideal** supremum:

1. If $\mathcal A$ is free with exponents exactly $(1,d_1,d_2)$, then
   $S_{\lambda,\beta}(\mathcal A;d_1,d_2)=0$ for **every** $\lambda>0$.
2. If $\mathcal A$ is **not** free with exponents $(1,d_1,d_2)$ — this
   includes genuinely nonfree arrangements *and* free arrangements whose
   true exponent pair differs from $(d_1,d_2)$ — then there is a constant
   $C_{\mathcal A}>0$ (depending on $\mathcal A$ and the pair) with

   $$S_{\lambda,\beta}(\mathcal A;d_1,d_2)\;\ge\;
   \frac{\lambda}{C_{\mathcal A}\,2^{\,1-\beta}+\lambda}
   \;\xrightarrow[\lambda\to\infty]{}\;1,$$

   and in particular $1-S_{\lambda,\beta}\le C_{\mathcal A}2^{1-\beta}/\lambda
   = O(1/\lambda)$.

Throughout, $L_j := L_{\mathcal A,d_j}$, $K_j := \ker L_j = D(\mathcal A)_{d_j}$,
$R(u,v)=\|L_1u\|^2+\|L_2v\|^2$, $B(u,v)=\det M(\theta_E,u,v)$, and
$q_{\mathcal A}=Q_{\mathcal A}/\|Q_{\mathcal A}\|$.

## Preliminaries

**(P1) $\|L_j\|\le 1$, hence $R\le 2$ on the product of unit spheres.**
Each line block of $L_j$ is $u\mapsto(I-\Pi_{\alpha,d})\,u(\alpha)$ with
$\|\alpha\|=1$. Orthogonal projection is a contraction and, by
Cauchy–Schwarz over the three components,
$\|u(\alpha)\| = \|a_0f_0+a_1f_1+a_2f_2\| \le \big(\sum_i|a_i|^2\big)^{1/2}
\big(\sum_i\|f_i\|^2\big)^{1/2} = \|u\|$. With the $1/\sqrt n$ prefactor,
$\|L_j u\|^2 \le \frac1n\sum_{i=1}^n\|u\|^2 = \|u\|^2$. So on unit spheres
$R(u,v)\le 2$.

**(P2) Cauchy–Schwarz on the numerator.** $|\langle B,q_{\mathcal A}\rangle|^2
\le \|B\|^2\|q_{\mathcal A}\|^2=\|B\|^2$, hence always
$\Gamma_{\lambda,\beta}(u,v)\le \dfrac{\|B\|^2}{\|B\|^2+\lambda R^\beta}$.

**(P3) $B$ is a bounded bilinear map.** $E_{d_1}\times E_{d_2}\to S_n$ is
bilinear between finite-dimensional normed spaces, so
$M_B:=\sup_{\|u\|=\|v\|=1}\|B(u,v)\|<\infty$.

**(P4) Pseudoinverse constants.** $L_j$ is a linear map between
finite-dimensional inner-product spaces; define uniformly
$c_j:=\|L_j^\dagger\|$ (the Moore–Penrose pseudoinverse norm; when
$L_j\neq0$ this equals $1/\sigma^+_{\min}(L_j)$, and when $L_j=0$ the
standard convention $L_j^\dagger=0$ gives $c_j=0$). For the orthogonal
decomposition $u=u_K+u_\perp$ with $u_K\in K_j$, $u_\perp\in K_j^\perp$,
one has $\|u_\perp\|=\|L_j^\dagger L_j u\|\le c_j\|L_j u\|$; if $L_j=0$
then $K_j^\perp=\{0\}$, $u_\perp=0$, and the estimate holds trivially.
(If $L_j$ is injective, $K_j=\{0\}$ and $u_\perp=u$.)

## Step 1: $B(K_1,K_2)=\{0\}$ in the non-target-free case

Let $u\in K_1$, $v\in K_2$; these are exact logarithmic derivations of
degrees $d_1, d_2$. For every defining form $\alpha_i$, logarithmicity
means $u(\alpha_i),v(\alpha_i)\in(\alpha_i)$, and trivially
$\theta_E(\alpha_i)=\alpha_i\in(\alpha_i)$. Restricting to the plane
$\alpha_i=0$ in $\mathbb C^3$, the three coefficient rows of
$M(\theta_E,u,v)$ become everywhere tangent to that 2-dimensional plane,
hence linearly dependent along it, so $\alpha_i\mid B(u,v)$. As the
$\alpha_i$ are pairwise non-proportional, $Q_{\mathcal A}\mid B(u,v)$.
Now $B(u,v)\in S_n$ **and may be zero**: since $Q_{\mathcal A}$ divides
$B(u,v)$ and both lie in $S_n$, there is a scalar $c$ — possibly $c=0$ —
with

$$B(u,v)=c\,Q_{\mathcal A}.$$

If $c\neq 0$, Saito's criterion states that $\{\theta_E,u,v\}$ is a free
basis, making $\mathcal A$ free with exponents $(1,d_1,d_2)$ — contrary to
assumption. (For a free arrangement whose true exponents differ from
$(d_1,d_2)$, freeness with $(1,d_1,d_2)$ is likewise impossible: the
exponent multiset of a free module is an invariant of $D(\mathcal A)$.)
Hence $c=0$ and $B$ vanishes identically on $K_1\times K_2$. $\square$

## Step 2: the residual controls the determinant

Claim: with the **finite** constant
$C_{\mathcal A}:=3M_B^2\,(c_1^2+c_2^2+c_1^2c_2^2)\ge 0$
(positivity is unnecessary for the limit; $C_{\mathcal A}=0$ only makes the
bound stronger),

$$\|B(u,v)\|^2\;\le\;C_{\mathcal A}\,R(u,v)\qquad\text{on }\|u\|=\|v\|=1.$$

Decompose $u=u_K+u_\perp$, $v=v_K+v_\perp$ as in (P4). By Step 1 and
bilinearity,

$$B(u,v)=\underbrace{B(u_K,v_K)}_{=0}+B(u_K,v_\perp)+B(u_\perp,v_K)
+B(u_\perp,v_\perp),$$

so by (P3), $\|u_K\|,\|v_K\|\le 1$, and (P4):

$$\|B(u,v)\|\le M_B\big(\|v_\perp\|+\|u_\perp\|+\|u_\perp\|\|v_\perp\|\big)
\le M_B\big(c_2\|L_2v\|+c_1\|L_1u\|+c_1c_2\|L_1u\|\,\|L_2v\|\big).$$

Square, apply $(x+y+z)^2\le 3(x^2+y^2+z^2)$, and use
$\|L_1u\|^2,\|L_2v\|^2\le R$ together with $\|L_2v\|^2\le 1$ from (P1):

$$\|B\|^2\le 3M_B^2\big(c_2^2\|L_2v\|^2+c_1^2\|L_1u\|^2
+c_1^2c_2^2\|L_1u\|^2\underbrace{\|L_2v\|^2}_{\le 1}\big)
\le 3M_B^2\,(c_1^2+c_2^2+c_1^2c_2^2)\,R. \qquad\square$$

(The constant is deliberately crude; only existence matters for the limit.)

## Step 3: the limit

Fix $(u,v)$ on the unit spheres. If $R=0$ then $u\in K_1,v\in K_2$, so
$B=0$ by Step 1 and $\Gamma=0$ (the $0/0$ base-locus convention — the
numerator vanishes identically there). If $R>0$, combine (P2), Step 2, and
the monotonicity of $t\mapsto t/(t+s)$:

$$\Gamma_{\lambda,\beta}(u,v)\le\frac{\|B\|^2}{\|B\|^2+\lambda R^\beta}
\le\frac{C_{\mathcal A}R}{C_{\mathcal A}R+\lambda R^\beta}
=\frac{C_{\mathcal A}R^{1-\beta}}{C_{\mathcal A}R^{1-\beta}+\lambda}.$$

Since $0<\beta<1$ and $R\le 2$ by (P1), $R^{1-\beta}\le 2^{1-\beta}$, and
$x\mapsto x/(x+\lambda)$ is increasing:

$$\sup_{\|u\|=\|v\|=1}\Gamma_{\lambda,\beta}
\le\frac{C_{\mathcal A}2^{1-\beta}}{C_{\mathcal A}2^{1-\beta}+\lambda},
\qquad
S_{\lambda,\beta}\ge\frac{\lambda}{C_{\mathcal A}2^{1-\beta}+\lambda}
\xrightarrow[\lambda\to\infty]{}1,$$

with rate $1-S_{\lambda,\beta}\le C_{\mathcal A}2^{1-\beta}/\lambda$.
$\square$

Remark on $\beta=1$: the **hard-penalty limit itself also holds at
$\beta=1$** — Step 2 with $R^\beta=R$ gives directly
$\Gamma_{\lambda,1}\le C_{\mathcal A}/(C_{\mathcal A}+\lambda)$, hence
$S_{\lambda,1}\ge\lambda/(C_{\mathcal A}+\lambda)\to1$. What fails at
$\beta=1$ is the *near-base-locus continuity and compact-attainment*
analysis used elsewhere for the fixed-$\lambda$ theory ($\|B\|^2$ and the
penalty scale identically along $u\to K_1,v\to K_2$, so the $0/0$ locus is
no longer repelling), which is why the production functional keeps
$\beta<1$; for $0<\beta<1$ the near-kernel directions are harmless:
$\|B\|^2=O(R)$ while the penalty is $\lambda R^\beta\gg R$ as $R\to0$.

## The target-free control case

If $\mathcal A$ is free with exponents exactly $(1,d_1,d_2)$, pick an exact
Saito basis $\{\theta_E,\theta_1,\theta_2\}$ and normalize
$u^*=\theta_1/\|\theta_1\|$, $v^*=\theta_2/\|\theta_2\|$. Then
$u^*\in K_1$, $v^*\in K_2$ gives $R(u^*,v^*)=0$, and
$B(u^*,v^*)=c\,Q_{\mathcal A}$ with $c\neq0$, so the denominator is
$\|B\|^2>0$ and

$$\Gamma_{\lambda,\beta}(u^*,v^*)
=\frac{|c|^2\|Q\|^2\cdot\|q\|^2}{|c|^2\|Q\|^2}=1
\quad\text{for every }\lambda>0,$$

hence $S_{\lambda,\beta}(\mathcal A;d_1,d_2)=0$ for every $\lambda$. $\square$

## Scope of the numerical experiments

This is a theorem about the **ideal supremum**. A finite multistart returns
$\widehat\Gamma\le\sup\Gamma$, hence $\widehat S=1-\widehat\Gamma\ge S$.
Numerical sweeps therefore *validate the implementation* (free controls at
0 for all $\lambda$ when seeded with exact Saito pairs; common-candidate-
pool monotonicity, which is exact because each fixed candidate's
$\Gamma_\lambda$ is nonincreasing in $\lambda$; $O(1/\lambda)$ scaling of
$1-\widehat S$) — they do not, and cannot, prove the theorem. Conversely,
at large $\lambda$ even a poor optimizer returns values near 1, so
convergence of $\widehat S\to1$ alone measures nothing about optimizer
quality; the informative checks are the controls and the scaling. The
constant $C_{\mathcal A}$ computed from floating singular values
($c_j=1/\sigma^+_{\min}(L_j)$, sampled $M_B$) is a **diagnostic**, not a
certificate; a certified bound would require interval arithmetic or exact
rank/singular-value bounds.

Verified computationally by `benchmarks/hard_penalty_limit.py` and
`tests/test_hard_penalty_limit.py`; results under
`results_penalized_saito/<date>/hard_penalty/`.
