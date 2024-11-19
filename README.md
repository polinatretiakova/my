# Импортируем необходимые библиотеки
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import uvicorn  # Импортируем сервер uvicorn для запуска приложения

# Создаем экземпляр приложения FastAPI
app = FastAPI()

# Загружаем набор данных ирисов из библиотеки sklearn
iris = load_iris()
X = iris.data
y = iris.target

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Сохраняем обученную модель в файл для дальнейшего использования
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Загрузка модели из файла для предсказаний
with open('iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Определяем корневой маршрут ("/") для отображения формы ввода
@app.get("/")
async def index():
    return HTMLResponse(
        """
        <html>
        <head>
            <title>Классификация Ирисов</title>
        </head>
        <body>
            <h1>Классификация Ирисов</h1>
            <form method="post" action="/predict">
                <label for="sepal_length">Длина чашелистика:</label>
                <input type="number" id="sepal_length" name="sepal_length"><br><br>
                <label for="sepal_width">Ширина чашелистика:</label>
                <input type="number" id="sepal_width" name="sepal_width"><br><br>
                <label for="petal_length">Длина лепестка:</label>
                <input type="number" id="petal_length" name="petal_length"><br><br>
                <label for="petal_width">Ширина лепестка:</label>
                <input type="number" id="petal_width" name="petal_width"><br><br>
                <input type="submit" value="Отправить">
            </form>
        </body>
        </html>
        """
    )

# Определяем маршрут для обработки POST-запроса с данными формы
@app.post("/predict")
async def predict(sepal_length: float = Form(...),
                  sepal_width: float = Form(...),
                  petal_length: float = Form(...),
                  petal_width: float = Form(...)):
   
    # Формируем входные данные
    data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Используем загруженную модель для предсказания класса цветка
    prediction = loaded_model.predict(data)[0]

    # Задаем имена для каждого класса ириса
    iris_names = ['setosa', 'versicolor', 'virginica']
    predicted_flower = iris_names[prediction]

    # Возвращаем HTML-страницу с результатом предсказания
    return HTMLResponse(
        f"""
        <html>
        <head>
            <title>Результат</title>
        </head>
        <body>
            <h1>Результат классификации</h1>
            <p>Предсказанный цветок: {predicted_flower}</p>
        </body>
        </html>
        """
    )

# Проверяем, если этот файл исполняется как главный, запускаем сервер
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
