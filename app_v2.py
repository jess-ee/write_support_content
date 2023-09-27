
import os
import pprint
import streamlit as st 

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.utilities import WikipediaAPIWrapper 
from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper

apikey = os.getenv('OPENAI_API_KEY')
googleapikey = os.getenv('SERPER_API_KEY')

# App framework
st.title('Van klantvraag ➡️ ➡️ naar support artikel')
topic= st.text_input('Welke klantvraag kon de chatbot niet beantwoorden?') 

# Prompt template1
query_template = PromptTemplate(
    input_variables = ['topic'], 
    template="""Bedenk een goede google zoekopdracht om de klantvraag te kunnen beantwoorden 

                Onbeantwoorde klantvraag: {topic}
                Bijpassende zoekopdracht:"""
)

# Prompt template2 
article_template = PromptTemplate(
    input_variables = ['query', 'google_search_results'], 
    template="""Schrijf een  support artikel op basis van deze google zoekopdracht {query} en maak daarbij gebruik
                van de informatie die op het internet is gevonden: {google_search_results}. 

                Houdt het concreet en behulpzaam. 
                 
     """
)

# Prompt template3 
style_template = PromptTemplate(
    input_variables = ['article'], 
    template=""" Je bent een content specialist van elektronika verkoper Coolblue.
    Je gaat nu een artikel aanpassen naar de stijl van Coolblue. Je houdt daarbij rekening met de volgende stijl regels tussen deze scheidingstekens <<< >>>  

    <<< Schrijftips en -trucs
    Onze teksten zijn simpel en begrijpelijk. Moeilijke dingen maken we makkelijk. Want we hebben verstand van zaken en leggen dat op een eenvoudige manier uit. Of het nu over een nieuwe laptop gaat, of over een nieuwe functionaliteit op onze website.

    Hoe dan?
    Met deze tips kom je een heel eind:

    Bedenk vóór je begint met schrijven wat je boodschap is:
    Wat moet je lezer weten, voelen en doen?
    Wat boeit dat? Waarom is je boodschap belangrijk voor je lezer?
    Beschrijf maximaal één gedachte per zin en maximaal één thema per alinea. Zo voorkom je dat je ingewikkelde tekstbruggetjes moet verzinnen om de boel aan elkaar te schrijven.
    Probeer niet alle informatie in één zin te verwerken. Maak van lange én ingewikkelde zinnen 2 of meer kortere, beter gestructureerde zinnen.
    Wissel OPA-zinnen (onderwerp, persoonsvorm, andere zinsdelen) eens af met OAP-, POA-, PAO-, APO- of AOP-zinnen.
    Pas op met "maar" en "echter". Deze woorden introduceren een tegenstrijdigheid.
    Vermijd uitroeptekens, dat staat zo schreeuwerig!!!!!!! Stelregel: gebruik er niet meer dan 1 per stukje tekst.
    Neem jezelf niet heel serieus, maar de klant wel. Doe daarom echte, concrete beloftes. Dus niet "binnen 2 weken", maar "op 5 januari" 

    Kort en bondig
    Houd je teksten kort en bondig. Korter. Niemand heeft zin om een langdradige tekst te lezen. Schrijven is schrappen. Houd daarom deze tips in je achterhoofd:

    Kill your darlings. Ga na het schrijven van je tekst nog eens door de tekst heen. Voegt die ene zin die je zo leuk vindt écht iets toe? Vaak wordt je tekst beter als je juist zo’n zin schrapt.
    Wees zuinig met bijvoeglijke naamwoorden. Gebruik alleen treffende bijvoeglijke naamwoorden en laat de woorden die niets zeggen weg. Nietszeggende bijvoeglijke naamwoorden zijn: ideaal, multifunctioneel, nieuw, optimaal, perfect, snel, uniek.
    Schrijf actief. Vermijd hulpwerkwoorden als zijn, hebben, worden, zullen en kunnen. Die maken de tekst onpersoonlijk, moeilijk en saai.
    Voorkom herhalingen.

    >>>

    Vertaal nu het volgende artikel naar Coolblue stijl: {article}               
     """)


# Llms
model = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo-16k")
#llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0.9, max_tokens=3800) 
query_chain = LLMChain(llm=model, prompt=query_template, verbose=True, output_key='query')
article_chain = LLMChain(llm=model, prompt=article_template, verbose=True, output_key='article')
style_chain = LLMChain(llm=model, prompt=style_template, verbose=True, output_key='styled_article')

search = GoogleSerperAPIWrapper()


if st.button('Maak een nieuw support artikel!'):
    if topic: 
        query_response = query_chain.run(topic=topic)
        
        
        # Directly using the query_response
        query = query_response
        google_search_results = search.run(query) 
        
        if not google_search_results:
            st.write("No good Google Search Result was found.")
        else:
            article_response = article_chain.run(query=query, google_search_results=google_search_results)
            
            # Directly using the article_response
            article = article_response
            
            # New addition: Style the article
            styled_article_response = style_chain.run(article=article)
            styled_article = styled_article_response  # Assuming style_chain.run() returns a string
            
            st.header("Google zoekopdracht")
            st.write(query) 
            
            st.header("Nieuw support artikel")
            st.write(article)
            
            st.header("Artikel in Coolblue stijl")
            st.write(styled_article)

