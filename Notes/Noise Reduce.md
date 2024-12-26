#signal_processing
For noise reduce, work plan:
יש כמה סוגי רעשים בסיסיים שכדאי להכיר תאורטית (לפחות לדעת על קיומם ותכונותיהם הבסיסיות) לפני שמתחילים לעבוד:
1. White/Thermal/Static noise
2. Transient/Dynamic noise
3. Background/Additive noise
4. Reverberation (RIR convolution)
5. Distortion/Non-linear noise (e.g compression)

 ספריית הפייתון noisereduce שעושה spectral-gating ועובדת מצוין לרעשים סטטיים או טרנזיאנטים פשוטים
 חוץ מזה קיימים מגוון מודלים של denoising (להסרת additive noise), ויש מספר מודלי dereverberation (להסרת הדהוד), לכולם יש יתרונות חסרונות ובעיות, אני ממליץ ללמוד קצת על התחום לפני שימוש בהם, ספציפית על סוגי המודלים הבאים:
1. Spectral Mask denoisers
2. Neural Vocoders (e.g HifiGAN)
3. Waveform denoisers (e.g "denoiser" by Meta)
רוב המודלים מתקשים בהכללה, לכן אני ממליץ לבדוק אותם היטב.
אני ממליץ על CMGAN כמודל יחסית מודרני עם קוד סביר שמכליל במידה סבירה, לא יעיף אתכם מהכיסא אבל גם לא ילך לעזעזל ברגע שתזיזו אותו טיפונת מאזור הנוחות.


## White Noise
נחשב רעש לבן אם הממוצע שווה 0, standard deviation is constant, אין קורלציה בין הסדרת זמן לבין ה-Lag של הסדרת זמן. 
![[Pasted image 20241216174705.png]]
לדוגמא פה, הקורלציה בין הlags היא לא 0, יש התנהגות משוכפלת לאורך זמן ולכן לא רעש לבן.
ופה לדוגמא הstd לא קבוע עם הזמן 
![[Pasted image 20241216174846.png]]
Source -> https://www.youtube.com/watch?v=cr4zIXAmSRI&t=2s
White noise is not predictable by definition
אם אפשר להוכיח שהשארית היא רעש לבן או ממש קרובה לרעש לבן אז זה מודל טוב

## Thermal Noise -> Nyquist Noise
the motion of these charged particles and in particular the variation that you get because they're moving randomly creates fluctuation in density 
Source -> https://www.youtube.com/watch?v=RVTtHZ4IfZ0
![[Pasted image 20241216175639.png]]
התנועה של האלקטרונים יוצרת שינוי בצפיפות -> שינוי במתח בריזיסטור (נגד) 
## Dynamic Noise
משתנה עם הזמן, למשל רוח (או מקרופון שקולט רוח לפעמים)

סיכום מעניין https://www.analyticsvidhya.com/blog/2022/03/audio-denoiser-a-speech-enhancement-deep-learning-model/

## Noise Reduce Techniques
* [ ] LMS
* [ ] RLS
* [ ] Kalman
### noisereduce
#### Spectral Gating Algorithm
from time domain to frequency domain using FFT (fast Fourier transform)
 It works by computing a spectrogram of a signal (and optionally a noise signal) and estimating a noise threshold (or gate) for each frequency band of that signal/noise. That threshold is used to compute a mask, which gates noise below the frequency-varying threshold.