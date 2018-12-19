# Tensorflow TextCNN

## 参考

[Keras](https://morvanzhou.github.io/tutorials/machine-learning/keras/)

[Text CNN](http://www.tensorflownews.com/2018/04/06/%E4%BD%BF%E7%94%A8keras%E8%BF%9B%E8%A1%8C%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%9A%EF%BC%88%E4%B8%89%EF%BC%89%E4%BD%BF%E7%94%A8text-cnn%E5%A4%84%E7%90%86%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80/)

[Gensim入门教程](http://www.cnblogs.com/iloveai/p/gensim_tutorial.html)

## 结构说明

包含
1. 对面试者提问的gensim模型（未保存）与非gensim模型`textCNN_model_gensim.h5`
2. 接受面试者提问的gensim模型`textCNN_model_salary_gensim.h5`与非gensim模型`textCNN_model_salary.h5`

```
/tensorflow
	/data
		/input_total
			/question 	（问题）
			/answer		（回答）
			/salary_question （面试者问题）
	/scripts
		/models			（训练完成的keras模型）
		/string
			/module		（gensim预处理用到的字典等文件）
		test_textCNN.py 		（测试textCNN.py的脚本）
		test_textCNN_gensim.py 	（测试textCNN_gensim.py的脚本，缺少gensim版预处理）
		test_textCNN_mul.py 	（测试textCNN.py的批量处理脚本）
		textCNN.py 				（keras自带字典训练模型脚本）
		textCNN_gensim.py 		（去停词训练模型脚本，包含两种是否为gensim预处理函数）
```

其中，路径中提供的不同文件是根据预处理方法不同有所区别的

`stoplist`与`frequency`是两种预处理方法都需要用到的。

在非gensim处理中，需要用到：
- tokenizer(binary)

gensim处理中，需要用到：
- dictionary(.dict)
- dictionary(.mm & .mm.index)

最后在`/tensorflow/scripts/models`里生成.h5的模型文件。
