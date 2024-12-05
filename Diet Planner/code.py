# Step 1


code = """


import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.model_selection import train_test_split 

# Load the dataset
df = pd.read_csv("All_Diets.csv")

# Prepare the data for classification

X = df[['Protein(g)', 'Carbs(g)', 'Fat(g)', 'Cuisine_type']]
X = pd.get_dummies(X, columns=['Cuisine_type'], drop_first=True)
y_class = df['Diet_type']

# Prepare the data for regression (calorie prediction) 
y_reg = (df['Protein(g)'] * 4) + (df['Carbs(g)'] * 4) + (df['Fat(g)'] * 9) 

# Split the data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(   
    X, y_class, test_size=0.2, random_state=42
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42                            
)

# Train the models
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_class, y_train_class)

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Streamlit application 
st.title("Diet Planner")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Classify Diet", "Predict Calories", "Modify Data"])

# Tab 1: Classify Diet
with tab1:
    st.header("Classify Diet for a Recipe")
    protein = st.number_input("Protein (g):", min_value=0.0, step=0.1, key="protein_diet")
    carbs = st.number_input("Carbs (g):", min_value=0.0, step=0.1, key="carbs_diet")
    fat = st.number_input("Fat (g):", min_value=0.0, step=0.1, key="fat_diet")
    cuisine = st.selectbox("Cuisine Type:", df['Cuisine_type'].unique(), key="cuisine_diet")

    if st.button("Classify Diet"):
        input_data = pd.DataFrame(
            [[protein, carbs, fat, cuisine]],
            columns=['Protein(g)', 'Carbs(g)', 'Fat(g)', 'Cuisine_type']
        )
        input_data = pd.get_dummies(input_data, columns=['Cuisine_type'], drop_first=True)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Predict Diet Type
        predicted_diet = classifier.predict(input_data)[0]

        # Find Closest Match for Dish Name
        df['Distance'] = ((df['Protein(g)'] - protein) ** 2 +
                          (df['Carbs(g)'] - carbs) ** 2 +
                          (df['Fat(g)'] - fat) ** 2).apply(lambda x: x ** 0.5)
        closest_match = df[df['Diet_type'] == predicted_diet].sort_values('Distance').iloc[0]
        predicted_dish_name = closest_match['Recipe_name']

        # Nutritional information for the closest match dish

        closest_match_nutrition = closest_match[['Protein(g)', 'Carbs(g)', 'Fat(g)']]

        # Check for and exclude rows with any NaN or empty values
        closest_match_nutrition = closest_match_nutrition.dropna()

        # Styling the dish name
        st.markdown(f"<h2 style='color: #4CAF50; text-align: center;'>{predicted_dish_name}</h2>", unsafe_allow_html=True)

        st.success(f"Predicted Diet Type: {predicted_diet}")

        # Display nutritional information in a table
        st.subheader("Nutritional Information:")
        st.table(closest_match_nutrition)

# Tab 2: Predict Calories
with tab2:
    st.header("Predict Calories for a Recipe")
    protein = st.number_input("Protein (g) (Calories Tab):", min_value=0.0, step=0.1, key="protein_cal")
    carbs = st.number_input("Carbs (g) (Calories Tab):", min_value=0.0, step=0.1, key="carbs_cal")
    fat = st.number_input("Fat (g) (Calories Tab):", min_value=0.0, step=0.1, key="fat_cal")
    cuisine = st.selectbox("Cuisine Type (Calories Tab):", df['Cuisine_type'].unique(), key="cuisine_cal")

    if st.button("Predict Calories"):
        input_data = pd.DataFrame(
            [[protein, carbs, fat, cuisine]],
            columns=['Protein(g)', 'Carbs(g)', 'Fat(g)', 'Cuisine_type']
        )
        input_data = pd.get_dummies(input_data, columns=['Cuisine_type'], drop_first=True)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        predicted_calories = regressor.predict(input_data)[0]
        st.success(f"Predicted Calories: {predicted_calories:.2f}")

# Tab 3: Modify Data
with tab3:
    st.header("Modify the Dataset")
    st.write("Here is a preview of your dataset:")
    st.dataframe(df)

    if st.checkbox("Add a New Recipe"):
        new_protein = st.number_input("Protein (g):", min_value=0.0, step=0.1, key="new_protein")
        new_carbs = st.number_input("Carbs (g):", min_value=0.0, step=0.1, key="new_carbs")
        new_fat = st.number_input("Fat (g):", min_value=0.0, step=0.1, key="new_fat")
        new_cuisine = st.selectbox("Cuisine Type:", df['Cuisine_type'].unique(), key="new_cuisine")
        new_diet = st.text_input("Diet Type:", key="new_diet")

        if st.button("Add Recipe"):
            new_row = {
                'Protein(g)': new_protein,
                'Carbs(g)': new_carbs,
                'Fat(g)': new_fat,
                'Cuisine_type': new_cuisine,
                'Diet_type': new_diet
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("New recipe added!")
            st.dataframe(df)




"""

# Save the code to app.py
with open('app.py', 'w') as file:
    file.write(code)
print("app.py has been saved!")


# Step 2

# !pip install streamlit pyngrok

# Step 3

 # !ngrok config add-authtoken 2pA7bUG8itMEnmDJQXf84Zw81wU_3pxwefcBsUvHp6H3xLir1

 # Step 4

from pyngrok import ngrok
import subprocess

# Run Streamlit in the background
process = subprocess.Popen(['streamlit', 'run', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Create a public URL using ngrok
public_url = ngrok.connect(8502)
print(f"Streamlit app is live at: {public_url}")





