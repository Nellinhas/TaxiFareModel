# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
    
        # create distance pipeline
        distance = Pipeline([('distance', DistanceTransformer()),('scaler', StandardScaler())])
        
        # create time pipeline
        time = Pipeline([('time', TimeFeaturesEncoder('pickup_datetime')),('encoder', OneHotEncoder())])
        
        # create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('distance', distance, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']),
            ('time', time, ['pickup_datetime'])])
        
        # Add the model of your choice to the pipeline
        self.pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('linear_regression', LinearRegression())])
        
        return self.pipeline

    def run(self, X_train, y_train):
        """set and train the pipeline"""
        self.pipeline = self.pipeline.fit(X_train,y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    features = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    X = df[features]
    y = df.fare_amount
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = Trainer(X,y)
    model.set_pipeline()
    model.run(X_train,y_train)
    results = model.evaluate(X_test,y_test)
    print(results)
