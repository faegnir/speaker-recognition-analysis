const recordBtn = document.querySelector(".record");
const predictBtn = document.querySelector(".predict");
const clearBtn = document.querySelector(".clear");

const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('audioname');
const audioFilenameInput = document.getElementById('audio_name');

let recording = false;

function ExecPythonCommand(pythonCommand){
  var request = new XMLHttpRequest()
  request.open("GET", "/" + pythonCommand, true)
  request.send()
}

function startRecording() {
  ExecPythonCommand('mic_record/start');
  recordBtn.classList.add("recording");
  recordBtn.querySelector("p").innerHTML = "Listening...";
  recording = true;
}

function stopRecording() {
  ExecPythonCommand('mic_record/stop');
  recordBtn.querySelector("p").innerHTML = "Start Listening";
  recordBtn.classList.remove("recording");
  recording = false;
  var checkInterval = setInterval(function() {
    var request = new XMLHttpRequest();
    request.open("GET", "/get_last_recorded_filename", true);
    request.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
        var fileName = this.responseText;
        if (fileName) {
          fileNameDisplay.textContent = "File: " + fileName;
          audioFilenameInput.value = fileName;
          clearInterval(checkInterval);

        }
      }
    };
    request.send();
  }, 500);
  predictBtn.disabled = false;
}

recordBtn.addEventListener("click", () => {
  if (!recording) {
    startRecording();
  } else {
    stopRecording();
  }
});

fileInput.addEventListener('change', function() {
  fileNameDisplay.textContent = 'File: ' + this.files[0].name;
  predictBtn.disabled = false;
  audioFilenameInput.value = this.files[0].name;
});

clearBtn.addEventListener("click", () => {
  fileNameDisplay.textContent = 'File: ';
  predictBtn.disabled = true;
  fileInput.value="";
});
