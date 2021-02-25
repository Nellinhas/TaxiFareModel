# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"
        self.myname = "Alexandre Canellas"
        self.experiment_name = f"TaxifareModel_{self.myname}"

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

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.pipeline.fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        self.mlflow_log_metric('rmse', compute_rmse(y_pred, y_test))
        return compute_rmse(y_pred, y_test)
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    features = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    X = df[features]
    y = df.fare_amount
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = Trainer(X_train,y_train)
    model.set_pipeline()
    model.run()
    results = model.evaluate(X_test,y_test)
    print(results)
