#**Harnessing NLP for Enhanced Maternal Care**#
**Overview**

_This is a prototype solution Our solution leverages NLP techniques to provide insightful summaries of women's feedback on childbirth experiences and interactions with the healthcare system. This NLP product is powered by OpenAI's models (Whisper for speech recognition and GPT-3.5 for text summarization) using OpeanAI API, application development supported by the LangChain framework, and the prototype is presented using the Streamlit framework._

**Features**
  * Flexible Input methods: speech recording, uploading existing audio files for transcription, or utilizing written feedback, our prototype offers different input options with accessibility in mind.
  * Seamless Transcription: Transforming audio inputs into text transcripts, our solution ensures access to raw data will not be lost, and offers a foundation for LLMs further analysis.
  * Comprehensive Summarization: By extracting key insights from the transcripts, our prototype generates detailed summaries, divided by topics, based on the needs of the healthcare specialists, at the same time identifies red flags concerning women's or child's wellbeing, for a potential quick response and intervention.
  * Output Formats: Users receive both a text file containing the audio transcript and a CSV file containing the summary, red flags, and extracted topics, facilitating comprehensive analysis and follow-up actions. Audio recording and a transcript can be used, if necessary, for human audition of the LLM output and prompt engineering for healthcare needs.

    
**Installation**

  * Clone the repository to your local machine.
  * Install the required dependencies using pip install -r requirements.txt.
  * Ensure you have the necessary environment variables set up, including OpenAI API key in a separate .env file: OPENAI_API_KEY='...'

**Usage**

   * Run the Streamlit app using streamlit run app.py.
   * Choose the desired input method (speech recording, upload audio file, or written feedback).
   * View the generated summaries, red flags, and extracted topics in the output files provided.
