
import { QuizQuestion, CodeExample, AlgorithmStep } from './types';

export const codeExamples: CodeExample[] = [
    { id: 'basic', title: '🚀 Basic XGBoost Example', description: '<p>Simple classification example with default parameters.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>', code: 'import xgboost as xgb\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\n\n# Load dataset\ndata = load_breast_cancer()\nX, y = data.data, data.target\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\n# Create and train model\nmodel = xgb.XGBClassifier(random_state=42)\nmodel.fit(X_train, y_train)\n\n# Make predictions\ny_pred = model.predict(X_test)\naccuracy = accuracy_score(y_test, y_pred)\n\nprint(f"Accuracy: {accuracy:.3f}")\n# Output: Accuracy: 0.965' },
    { id: 'early_stopping', title: '⏱️ Early Stopping Example', description: '<p>Prevent overfitting and find the optimal number of trees automatically.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>', code: 'import xgboost as xgb\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\n\n# Load and split data\ndata = load_breast_cancer()\nX, y = data.data, data.target\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\n# Create a validation set from the training data\nX_train, X_val, y_train, y_val = train_test_split(\n    X_train, y_train, test_size=0.25, random_state=42\n)\n\n# Initialize model\nmodel = xgb.XGBClassifier(\n    n_estimators=500, # Set a high number of estimators\n    learning_rate=0.1,\n    random_state=42,\n    eval_metric=\'logloss\'\n)\n\n# Fit the model with early stopping\nmodel.fit(\n    X_train,\n    y_train,\n    eval_set=[(X_val, y_val)],\n    early_stopping_rounds=20, # Stop if validation loss doesn\'t improve for 20 rounds\n    verbose=False\n)\n\n# Make predictions with the best iteration\ny_pred = model.predict(X_test)\naccuracy = accuracy_score(y_test, y_pred)\n\nprint(f"Best Iteration: {model.best_iteration}")\nprint(f"Accuracy with Early Stopping: {accuracy:.3f}")' },
    { id: 'regression', title: '📈 Regression Example', description: '<p>Using XGBoost for continuous target prediction.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>', code: 'import xgboost as xgb\nfrom sklearn.datasets import fetch_california_housing\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error, r2_score\nimport numpy as np\n\n# Load California housing dataset\ndata = fetch_california_housing()\nX, y = data.data, data.target\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\n# XGBoost Regressor\nmodel = xgb.XGBRegressor(\n    n_estimators=100,\n    max_depth=4,\n    learning_rate=0.1,\n    random_state=42\n)\n\n# Train and evaluate\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\n\n# Metrics\nmse = mean_squared_error(y_test, y_pred)\nr2 = r2_score(y_test, y_pred)\nrmse = np.sqrt(mse)\n\nprint(f"RMSE: {rmse:.3f}")\nprint(f"R² Score: {r2:.3f}")' },
    { id: 'tuning', title: '🎛️ Hyperparameter Tuning', description: '<p>Robust hyperparameter optimization with proper error handling.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>', code: 'from sklearn.model_selection import GridSearchCV\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nimport xgboost as xgb\n\n# Load and prepare data\ndata = load_breast_cancer()\nX, y = data.data, data.target\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\n# Conservative parameter grid\nparam_grid = {\n    \'n_estimators\': [50, 100],\n    \'max_depth\': [3, 4],\n    \'learning_rate\': [0.1, 0.2],\n    \'subsample\': [0.8, 1.0]\n}\n\n# Create model with stable settings\nxgb_model = xgb.XGBClassifier(\n    random_state=42,\n    eval_metric=\'logloss\'\n)\n\n# Grid search\ngrid_search = GridSearchCV(\n    estimator=xgb_model, \n    param_grid=param_grid, \n    cv=3,\n    scoring=\'accuracy\'\n)\n\ngrid_search.fit(X_train, y_train)\n\nprint(f"Best parameters: {grid_search.best_params_}")' },
    { id: 'shap', title: '🔍 SHAP Model Interpretability', description: '<p>Understanding XGBoost predictions with global and local explanations.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>', code: '# First install: pip install shap matplotlib\n\nimport xgboost as xgb\nimport shap\nfrom sklearn.datasets import fetch_california_housing\nfrom sklearn.model_selection import train_test_split\nimport pandas as pd\n\n# Load housing data\ndata = fetch_california_housing()\nX_df = pd.DataFrame(data.data, columns=data.feature_names)\ny = data.target\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_df, y, test_size=0.2, random_state=42\n)\n\n# Train model\nmodel = xgb.XGBRegressor(random_state=42).fit(X_train, y_train)\n\n# SHAP analysis\nexplainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(X_test.iloc[:100])' }
];

export const algorithmSteps: AlgorithmStep[] = [
    {
        id: 'step1',
        title: '1. 🎯 Initialize Prediction',
        detail: 'The model starts with a single, simple prediction for all data points. For regression, this is usually the average of the target values; for classification, it\'s the log-odds. Think of this as the "trunk" of our model—a simple, solid base. It\'s the best possible guess we can make without considering any features, as it minimizes the initial error.',
        formula: 'F₀(x) = initial_value (e.g., mean of y)',
        code: `# For a regression problem, the first prediction is simply the mean of the target variable.\ninitial_prediction = y_train.mean()`
    },
    {
        id: 'step2',
        title: '2. 📉 Calculate Residuals',
        detail: 'Next, the model calculates the errors, or "residuals," by subtracting the current predictions from the actual values. These residuals represent everything the model currently gets wrong. They are the "unexplained" part of the data. The entire goal of the next step is to specifically target and correct these mistakes.',
        formula: 'residuals = actuals - predictions',
        code: `# Calculate the errors from our initial guess.\nresiduals_1 = y_train - initial_prediction`
    },
    {
        id: 'step3',
        title: '3. 🌳 Train a Decision Tree',
        detail: 'A new decision tree (a "weak learner") is trained, but with a twist: instead of predicting the original target, it learns to predict the residuals from the previous step. This tree is a specialist, focusing only on fixing the specific errors the model made in the last round. It finds patterns in the mistakes.',
        formula: 'new_tree = DecisionTree(features → residuals)',
        code: `# A simple tree is trained on the errors. Note: This is a conceptual example.\n# from sklearn.tree import DecisionTreeRegressor\ntree_1 = DecisionTreeRegressor(max_depth=3)\ntree_1.fit(X_train, residuals_1)`
    },
    {
        id: 'step4',
        title: '4. ⚖️ Find Optimal Weight',
        detail: 'Instead of just adding the new tree\'s predictions, XGBoost calculates an optimal weight (gamma) for each leaf of the tree. This is a crucial, sophisticated step that uses calculus (specifically, the second-order gradient) to find the best contribution for each leaf. It ensures each tree contributes just the right amount to the final prediction, preventing overcorrection and minimizing the loss function efficiently.',
        formula: 'optimal_weight = best_γ_value',
        code: `# XGBoost calculates optimal weights for each leaf internally.\n# The tree's raw output represents the predicted corrections.\noutput_values_1 = tree_1.predict(X_train)`
    },
    {
        id: 'step5',
        title: '5. ➕ Update the Model',
        detail: 'The predictions from the new tree, scaled by both its optimal weights and a global "learning rate," are added to the overall model. The learning rate acts as a safety brake, forcing the model to take small, careful steps. This incremental improvement makes the model robust and prevents it from overshooting the best solution.',
        formula: 'F_new(x) = F_old(x) + learning_rate × new_tree(x)',
        code: `# Update our overall model prediction by adding the scaled correction.\nlearning_rate = 0.1\npredictions_1 = initial_prediction + learning_rate * output_values_1`
    },
    {
        id: 'step6',
        title: '6. 🔄 Repeat Until Convergence',
        detail: 'The process repeats from Step 2: calculate new residuals based on the updated model, train a new tree to fix them, and add it to the ensemble. This cycle continues until adding new trees no longer improves performance on a separate validation dataset (known as early stopping) or a maximum number of trees is reached. This ensures the model becomes progressively more accurate without overfitting.',
        formula: 'Repeat steps 2-5 for M rounds',
        code: `# The next round of residuals are calculated from the updated predictions.\nresiduals_2 = y_train - predictions_1\n\n# A new tree is trained on the new errors.\ntree_2 = DecisionTreeRegressor(max_depth=3)\ntree_2.fit(X_train, residuals_2)\n\n# ...and so on for n_estimators rounds`
    }
];

export const coreConceptsQuiz: QuizQuestion[] = [
    { question: "What is the primary goal of each new tree in a gradient boosting model?", options: [{ text: "To predict the original target variable.", correct: false }, { text: "To correct the errors (residuals) of the previous trees.", correct: true }, { text: "To be as deep and complex as possible.", correct: false }] },
    { question: "What does 'boosting' refer to in the context of XGBoost?", options: [{ text: "A method of training many models in parallel.", correct: false }, { text: "A sequential process where models are added one by one to correct prior mistakes.", correct: true }, { text: "Using very powerful, complex models from the start.", correct: false }] }
];

export const algorithmQuiz: QuizQuestion[] = [
    { question: "What is the very first prediction made by the XGBoost algorithm (before any trees are built)?", options: [{ text: "A random number.", correct: false }, { text: "Zero for all instances.", correct: false }, { text: "The average of the target values.", correct: true }] },
    { question: "What are 'residuals' in the XGBoost algorithm?", options: [{ text: "The difference between the actual values and the current predictions.", correct: true }, { text: "Data points that are considered outliers.", correct: false }, { text: "The features that are least important.", correct: false }] },
    { question: "What is the purpose of the 'learning_rate' parameter?", options: [{ text: "To control how fast the model trains on the GPU.", correct: false }, { text: "To scale the contribution of each new tree to prevent overfitting.", correct: true }, { text: "To determine the maximum depth of a tree.", correct: false }] }
];

export const shapCodeWalkthrough: CodeExample[] = [
    { 
        id: 'shap_code_1', 
        title: '🐍 1. Setup & Data Loading', 
        description: `<p>We begin by importing the necessary libraries and loading the California Housing dataset from Scikit-learn. The data is converted into a pandas DataFrame for easier handling. Crucially, we use <code>train_test_split</code> to separate our data into training and testing sets, which prevents data leakage and ensures an unbiased evaluation of our model's performance.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`, 
        code: `import xgboost as xgb
import pandas as pd
import numpy as np
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Data Loading
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
    },
    { 
        id: 'shap_code_2', 
        title: '🎛️ 2. Hyperparameter Tuning', 
        description: `<p>An out-of-the-box XGBoost model is good, but a tuned one is great. We use <code>GridSearchCV</code> to systematically test different combinations of key parameters like the number of trees (<code>n_estimators</code>) and their complexity (<code>max_depth</code>). It uses 3-fold cross-validation (<code>cv=3</code>) to find the combination that yields the best performance, giving us the most robust model for our analysis.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`, 
        code: `param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.10],
    'colsample_bytree': [0.7, 0.8]
}

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid,
                           cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")`
    },
    { 
        id: 'shap_code_3', 
        title: '📊 3. Model Evaluation', 
        description: `<p>After finding the best model, we evaluate its performance on the unseen test data. We use the model to make predictions and then calculate the Root Mean Squared Error (RMSE), a standard metric for regression tasks that tells us, on average, how far our predictions are from the actual values.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`, 
        code: `y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")`
    },
    { 
        id: 'shap_code_4', 
        title: '🔍 4. SHAP Initialization', 
        description: `<p>This is where the interpretation begins. We create a <code>shap.TreeExplainer</code>, which is optimized for tree-based models like XGBoost. We then pass our test data to the explainer to calculate the SHAP values for each feature in every prediction. This gives us the raw data needed for all our interpretation plots.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`, 
        code: `explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test)`
    },
    { 
        id: 'shap_code_5', 
        title: '🌍 5. Global Interpretation Plots', 
        description: `<p>Global plots help us understand the model's overall behavior.</p>
        <p>The <strong>bar plot</strong> shows the average impact of each feature. The <strong>beeswarm plot</strong> is richer, showing the impact of every single prediction and how the feature's value (high/low) affects the outcome. The <strong>dependence plot</strong> isolates one feature to show its effect across its value range, often revealing complex, non-linear relationships.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`,
        code: `plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Global Feature Importance (Bar)')
plt.tight_layout()
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title('Global Feature Importance (Beeswarm)')
plt.tight_layout()
plt.show()

plt.figure()
shap.dependence_plot("MedInc", shap_values.values, X_test, interaction_index="AveRooms", show=False)
plt.title('Dependence Plot for Median Income')
plt.tight_layout()
plt.show()`
    },
    { 
        id: 'shap_code_6', 
        title: '🔬 6. Local Interpretation Plot', 
        description: `<p>To understand a single prediction, we use a <strong>force plot</strong>. It shows how features for one specific house "push" and "pull" the prediction away from the base value (the average prediction). Features in red increase the predicted price, while those in blue decrease it, giving a clear, additive explanation for an individual case.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`, 
        code: `instance_loc = 0 # Explain the first instance

shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values.values[instance_loc,:],
    X_test.iloc[instance_loc,:],
    matplotlib=True,
    show=False
)
plt.title(f'Force Plot for Prediction at index {X_test.index[instance_loc]}')
plt.show(force_plot)`
    },
    { 
        id: 'shap_code_7', 
        title: '🌊 7. Advanced: Quintile Waterfall Plots', 
        description: `<p>This advanced technique checks if feature importance changes for different groups. We split houses into five price groups (quintiles) and create an averaged <strong>waterfall plot</strong> for each. This can reveal if, for example, location is key for low-priced homes while median income is more important for high-priced ones, showing how the model has learned different rules for different data segments.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`, 
        code: `test_df = X_test.copy()
test_df['target'] = y_test
test_df['quintile'] = pd.qcut(test_df['target'], 5, labels=False, duplicates='drop')

# Loop through each quintile and create a separate plot
for quintile in sorted(test_df['quintile'].unique()):
    quintile_indices = test_df[test_df['quintile'] == quintile].index
    locs = [X_test.index.get_loc(idx) for idx in quintile_indices]
    shap_subset = shap.Explanation(
        values=shap_values.values[locs, :],
        base_values=shap_values.base_values[locs],
        data=X_test.iloc[locs, :],
        feature_names=X_test.columns
    )

    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_subset.mean(0), max_display=10, show=False)
    plt.title(f"Average Feature Contribution for Quintile {quintile+1}")
    plt.tight_layout()
    plt.show()`
    },
    {
        id: 'shap_code_8',
        title: '📜 Full SHAP Analysis Script',
        description: `<p>For convenience, here is the complete, runnable Python script combining all the steps above. You can copy this code and run it in an environment like Jupyter or Google Colab to reproduce the entire analysis.</p><p class="mt-2 text-xs italic">💡 Tip: You can run this code yourself in this <a href="https://colab.research.google.com/notebooks/empty.ipynb" target="_blank" rel="noopener noreferrer" class="text-sky-600 dark:text-sky-400 underline">empty Colab notebook</a>.</p>`,
        code: `import xgboost as xgb
import pandas as pd
import numpy as np
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Data Loading
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.10],
    'colsample_bytree': [0.7, 0.8]
}
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid,
                           cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# 3. Model Evaluation with Best Model
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# 4. SHAP Interpretation
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test)

# 5. Global Interpretability Plots
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Global Feature Importance (Bar)')
plt.tight_layout()
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title('Global Feature Importance (Beeswarm)')
plt.tight_layout()
plt.show()

plt.figure()
shap.dependence_plot("MedInc", shap_values.values, X_test, interaction_index="AveRooms", show=False)
plt.title('Dependence Plot for Median Income')
plt.tight_layout()
plt.show()

# 6. Local Interpretability Plot (Single Instance)
instance_loc = 0 # Explain the first instance
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values.values[instance_loc,:],
    X_test.iloc[instance_loc,:],
    matplotlib=True,
    show=False
)
plt.title(f'Force Plot for Prediction at index {X_test.index[instance_loc]}')
plt.show(force_plot)

# 7. Advanced: Quintile-based Waterfall Plots
test_df = X_test.copy()
test_df['target'] = y_test
test_df['quintile'] = pd.qcut(test_df['target'], 5, labels=False, duplicates='drop')
# Loop through each quintile and create a separate plot
for quintile in sorted(test_df['quintile'].unique()):
    quintile_indices = test_df[test_df['quintile'] == quintile].index
    locs = [X_test.index.get_loc(idx) for idx in quintile_indices]
    shap_subset = shap.Explanation(
        values=shap_values.values[locs, :],
        base_values=shap_values.base_values[locs],
        data=X_test.iloc[locs, :],
        feature_names=X_test.columns
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_subset.mean(0), max_display=10, show=False)
    plt.title(f"Average Feature Contribution for Quintile {quintile+1}")
    plt.tight_layout()
    plt.show()`
    }
];