import google.generativeai as genai

genai.configure(api_key="AIzaSyAwmZkp5bTPl6fpSKs6lMtC3BxfcOkPd84")
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Say hello")
print(response.text)