from flask import Flask
from sklearn.externals import joblib
from flask_restplus import Api,fields,Resource
from flask_api import FlaskAPI

#%%

app = Flask(__name__)

api = Api(
   app, 
   version='1.0', 
   title='Credit Loan Default System',
   description='A Loan Approval System Using Machine Learning API')

ns = api.namespace('Check Loan Request', 
   description='Approve Credit Operations')


resource_fields = api.model('Resource', {
    'result': fields.String,
})

parser = api.parser()
parser.add_argument(
   'RevolvingUtilizationOfUnsecuredLines', 
   type=float, 
   required=True, 
   location='form')
parser.add_argument(
   'age', 
   type=float, 
   required=True, 
   help='Age in years',
   location='form')
parser.add_argument(
   'NumberOfTime30-59DaysPastDueNotWorse', 
   type=float, 
   required=True, 
   location='form')
parser.add_argument(
   'DebtRatio', 
   type=float, 
   required=True, 
   location='form')
parser.add_argument(
   'MonthlyIncome', 
   type=float, 
   required=True, 
   help='Monthly income',
   location='form')
parser.add_argument(
   'NumberOfOpenCreditLinesAndLoans', 
   type=float, 
   required=True, 
   location='form')
parser.add_argument(
   'NumberOfTimes90DaysLate', 
   type=float, 
   required=True, 
   location='form')
parser.add_argument(
   'NumberRealEstateLoansOrLines', 
   type=float, 
   required=True, 
   location='form')
parser.add_argument(
   'NumberOfTime60-89DaysPastDueNotWorse', 
   type=float, 
   required=True, 
   location='form')
parser.add_argument(
   'NumberOfDependents', 
   type=float, 
   required=True, 
   help='Number of dependents',
   location='form')

@ns.route('/')
class CreditApi(Resource):

   @api.doc(parser=parser)
   @api.marshal_with(resource_fields)
   def post(self):
     args = parser.parse_args()
     result = self.get_result(args)

     return result, 201

   def get_result(self, args):
      debtRatio = args["DebtRatio"]
      monthlyIncome = args["MonthlyIncome"]
      dependents = args["NumberOfDependents"]
      openCreditLinesAndLoans = args["NumberOfOpenCreditLinesAndLoans"]
      pastDue30Days = args["NumberOfTime30-59DaysPastDueNotWorse"]
      pastDue60Days = args["NumberOfTime60-89DaysPastDueNotWorse"]
      pastDue90Days = args["NumberOfTimes90DaysLate"]
      realEstateLoansOrLines = args["NumberRealEstateLoansOrLines"]
      unsecuredLines = args["RevolvingUtilizationOfUnsecuredLines"]
      age = args["age"] 

      from pandas import DataFrame
      df = DataFrame([[
         debtRatio,
         monthlyIncome,
         dependents,
         openCreditLinesAndLoans,
         pastDue30Days,
         pastDue60Days,
         pastDue90Days,
         realEstateLoansOrLines,
         unsecuredLines,
         age
      ]])



      from sklearn.externals import joblib
      clf = joblib.load('model/loan_default4.pkl');

      result = clf.predict(df)
      if(result[0] == 1.0): 
         result = "Loan Denied" 
      else: 
         result = "Loan approved"

      return {
         "result": result
      }




if __name__ == '__main__':
    app.run(debug=True)
