# K近邻 （KNN）

K近邻算法K-Nearest Neighbors（KNN），K表示最近的K个样本。

![](https://www.kdnuggets.com/wp-content/uploads/popular-knn-metrics-0.png)

KNN 的基本思想是样本距离只够接近，样本的类型可以划分为一类。

> [!Note]
>
> K 值的取值一般基于经验。

对于 $n$ 维向量 $x$ 其距离公式为，欧拉距离。
$$
\sqrt{\sum_{i=1}^n\left(x_i^{(a)}-x_i^{(b)} \right)^2}
$$

> [!attention]
>
> KNN算法是一个不需要训练过程的算法，可以认为KNN算法发的模型就是全部训练数据本身。

```python
raw_data_x = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

x_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)

x = np.array([8.093607318, 3.365731514])

from sklearn.neighbors import KNeighborsClassifier
kNN_classifier = KNeighborsClassifier(n_neighbors=6)
kNN_classifier.fit(x_train, y_train)
x_predict = x.reshape(1, -1)
y_predict = kNN_classifier.predict(x_predict)
print(predict_y)
```

