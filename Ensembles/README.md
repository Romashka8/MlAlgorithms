# ✅ Idea

Main idea/reason for models ensebling is bias-variance decomposition:
$$Q(a)=\mathbb{E}_x​bias_X^2​a(x,X)+\mathbb{E}_x​V_X​[a(x,X)]+σ^2$$
Where:
**Variance**:
$$\mathbb{V}_X​[a(x,X)]=E_X​[a(x,X)−E_X​[a(x,X)]]^2$$
**Bias**:
$$bias_X​a(x,X)=f(x)−E_X​[a(x,X)]$$
**Noise**:
$$σ^2=E_x​E_ϵ​[y(x,ϵ)−f(x)]^2$$
**More info**: https://education.yandex.ru/handbook/ml/article/ansambli-v-mashinnom-obuchenii