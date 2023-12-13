# Union of Convex Separators
A novel machine learning algorithm for binary classification problem in the making. To appear in CODS COMAD 2024 (https://cods-comad.in/).

Designed by : Amit Prasad (https://www.linkedin.com/in/amit-prasad-951081119/), Prof. Rahul Garg(https://www.cse.iitd.ac.in/~rahulgarg/), Prof. Yogish Sabharwal (https://www.cse.iitd.ac.in/~yogish/HomePage/Home.html)

A Union of Convex Separator classifies one class from the other by learning convex separators for convexly separable subsets of the class. Something like this where the blue colored points are one class and orange colored is another.
![ucs_syn_datasets drawio](https://github.com/Amit-Prasad/union_of_convex_separators/assets/22973646/8b55b491-f695-4b9a-9aa7-7edea87566e2)

Following are the images showing the individual convex separators that form the union.
![image](https://github.com/Amit-Prasad/union_of_convex_separators/assets/22973646/b7a5f27f-300a-418b-a887-6b7e6596db6c)
![image](https://github.com/Amit-Prasad/union_of_convex_separators/assets/22973646/2591818a-2d91-4bd5-a8de-c85f8d1c94f5)
For evaluation on synthetic data:
Run the file ucs.py directly

The performance on real datasets is shown here:
| Algorithm                 |                      AUROC Scores on datasets                                |
| ---------                 | ------|---------------------|----------|------------|----------|-------------|
| ---------                 | Churn | Covertype (Subset)  | Diabetes | Ionosphere | Shoppers | Telco Churn |
| Union of Convex Separators| 0.85  | 0.88                | 0.86     | 0.96       |0.91      | 0.85        |
| Logistic Regression       | 0.75  | 0.85                | 0.79     | 0.91       |0.89      | 0.85        |
| MLP 2                     | 0.86  | 0.89                | 0.85     | 0.94       |0.91      | 0.62        |
| MLP 3                     | 0.85  | 0.91                | 0.85     | 0.98       |0.91      | 0.82        |
| Random Forest             | 0.82  | 0.91                | 0.85     | 0.97       |0.92      | 0.81        |
| MLP 1                     | 0.86  | 0.89                | 0.86     | 0.99       |0.92      | 0.81        |
| XGBoost                   | 0.83  | 0.91                | 0.8      | 0.93       |0.92      | 0.8         |
      

For evaluation on real datasets:
Run ucs_run.py

Make sure to specify file paths correctly.

Read the full paper union_of_convex_separator.pdf
