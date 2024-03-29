function startDictation() {
    //document.getElementById('transcript').value = "";
    if (window.hasOwnProperty('webkitSpeechRecognition')) {
      var recognition = new webkitSpeechRecognition();

      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.lang = 'en-US';
      recognition.start();

      recognition.onresult = function (e) {        
        document.getElementById('transcript').value = e.results[0][0].transcript;
        document.getElementById('dummy_inp').value = e.results[0][0].transcript;
        recognition.stop();
        console.log(document.getElementById('dummy_inp').value)
        //document.getElementById('speech_input').submit();
      };

      recognition.onerror = function (e) {
        recognition.stop();
      };
    }
  }