from dotenv import load_dotenv


from handlers.lemon import LemonMarketsAPI


load_dotenv()


def sentiment_analysis():
    lemon_api: LemonMarketsAPI = LemonMarketsAPI()
