# ✅ Idea

Main idea/reason for models ensebling is bias-variance decomposition:<br>
$$Q(a)=\mathbb{E}_x​bias_X^2​a(x,X)+\mathbb{E}_x​V_X​[a(x,X)]+σ^2$$<br>
Where:<br>
**Variance**:<br>
$$\mathbb{V}_X​[a(x,X)]=E_X​[a(x,X)−E_X​[a(x,X)]]^2$$<br>
**Bias**:<br>
$$bias_X​a(x,X)=f(x)−E_X​[a(x,X)]$$<br>
**Noise**:<br>
$$σ^2=E_x​E_ϵ​[y(x,ϵ)−f(x)]^2$$<br>
**More info**: https://education.yandex.ru/handbook/ml/article/ansambli-v-mashinnom-obuchenii
