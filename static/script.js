const recordBtn = document.querySelector(".record");
const btnCss = document.getElementById("rcbtn");
const predictBtn = document.querySelector(".predict");
const clearBtn = document.querySelector(".clear");


const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('audioname');
const audioFilenameInput = document.getElementById('audio_name');

let recording = false;
let timeoutId;

function ExecPythonCommand(pythonCommand){
  var request = new XMLHttpRequest()
  request.open("GET", "/" + pythonCommand, true)
  request.send()
}

async function startRecording() {
  ExecPythonCommand('mic_record/start');
  recordBtn.disabled=true;
  recordBtn.classList.add("recording");
  recordBtn.querySelector("p").innerHTML = "Listening...";
  recording = true;
  btnCss.style.backgroundColor = "#a5a5a5";
  const timeOut = (secs) => new Promise((res) => setTimeout(res, secs * 1000));
  await timeOut(3)
  recordBtn.disabled=false;
  btnCss.style.backgroundColor = "#e74135";
  timeoutId = setTimeout(() => {
    stopRecording();
  }, 30000);
}

function stopRecording() {
  clearTimeout(timeoutId);
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
