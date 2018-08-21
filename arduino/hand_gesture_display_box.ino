// Takes values of PIN_A, PIN_B and PIN_C of Google Vision AIY kit
// (1 if HIGH or 0 if LOW) 
// and calculates Signal = 4*A + 2*B + C which takes values between 0 and 7
// Based on the vaue of Signal a particular LED is lit:
// Signal=0 => No action (All LEDs are off)
// Signal=1 => Left
// Signal=2 => Right
// Signal=3 => Forward
// Signal=4 => Backward
// Signal=5 => Stop
// Signal=6 => Alert (All LEDs are blinking)
// Signal=7 => No action (All LEDs are off)

//Assign Arduino (input) pins - 
// pinA of Arduino corresponds to PIN_A or Google Vision AIY kit, 
// pinB - to PIN_B and pinC - to PIN_C

int PinA = 10;
int PinB = 11;
int PinC = 12;

// Assign Ardiuno (output) pins for LEDs 
int Left = 3;
int Right = 7;
int Forward = 6;
int Backward = 5;
int Stop = 4;

// Initiate variables
int A = 0;
int B = 0;
int C = 0;
int Signal = 0;

int hand_gestures[]={Left, Right, Forward, Backward, Stop};

// Alert
int alert(){
  for (int j=0; j<5; j++){
      digitalWrite(hand_gestures[j], HIGH);
  }
  delay(100);  
  for (int j=0; j<5; j++){
      digitalWrite(hand_gestures[j], LOW);
  }
  delay(100);
}

// Activate LED for a classified hand gesture 
int activate(int S){
  if (S == 6){
    alert();
  }
  else if (S > 5){
    S = 0;
  }
  for (int i=0; i<5; i++){
    if (S == (i+1)){
      digitalWrite(hand_gestures[i], HIGH);
    }
    else{
      digitalWrite(hand_gestures[i], LOW);
    }
    delay(100);
  }
}

void setup()
{
  pinMode(PinA, INPUT);
  pinMode(PinB, INPUT);
  pinMode(PinC, INPUT);

  for (int i=0; i<5; i++){
    pinMode(hand_gestures[i], OUTPUT);
  }
}

void loop()
{
  A = digitalRead(PinA);     // read the input pin A
  B = digitalRead(PinB);     // read the input pin B
  C = digitalRead(PinC);     // read the input pin C

  Signal = A*4 + B*2 + C;

  activate(Signal); 
}
