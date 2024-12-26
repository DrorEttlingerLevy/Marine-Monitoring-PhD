למידה מדוגמאות עבר בלי לתת דומגאות מפורשות, הכללה. אי אפשר לתכנת בתנאים פשוטים שאלה שמשתנה (כמו בתמונה לזהות מי עם משקפיים, התמונה שונה כל פעם)
לא כופים סטטיסטיקה מסויימת על הדתא (כמו גאוסיאנית למשל)
## KNN 
מודל KNN מחזיר את סה"כ הלייבל שהוא הרוב בסביבה (K) ליד הדוגמא
$$ \{ y_{\pi i}(x):i \leq k \}$$ $$ S = {(x_1,y_1),...(x_m,y_m)} $$
אנחנו לא יודעים איך התקבלה ההחלטה על בסיס הדוגמאות S מה הלייבל, **זה מה שמודל מנסה לחזות**
גבול החלטה - המקום שבו ההסתברות היא חצי בין שתי הדוגמאות השכנות, יכול להשתנות כתלות ב-K
ככל ש-K גדול יותר ככה גבול החלטה "חותך" יותר, פחות איים צפים
![[Pasted image 20241217065051.png]]
over fitting התאמה גבוהה לסט אימון ולא לדתא אמיתי
ככל שK גדול יותר ככה אנחנו עולים ברמת ההכללה
![[Pasted image 20241217065603.png]]
אם K קטן ממש, אז השגיאה עבור הסט אימון היא 0 (המודל צודק תמיד) אבל כשמכללים על דתא אמיתי מקבלים שגיאה גדולה. אם K=1, אז כל דגימה הכי קרובה לעצמה, אז ברור שאין טעות למודל
בtraning set עליו אתה מבצע את הלמידה, ו-test אתה עושה ולידציה 
אפשר להימנע מoverfitting ע"י רגוליזציה, איזון בין ביצועים למורכבות, רגוליזציה תורם ליציבות כלומר המודל פחות רגיש לשינויים קטנים. מה זה יציבות? עד כמה השתנתה הפרדיקציה בהתאם לשינויים בtraning set, אם היינו מוסיפים או משנים דוגמאות
**ב-KNN לא מניחים קשר לינארי או כל קשר אחר, רק בודקים לפי מרחק, לא מאלצים קשר בין הפרמטרים**
## Logistic Regression
בבריגרסיה **לינארית** אנחנו מאלצים קשר לינארי 
$h_s(x_i) = w_1x_1+...w_dx_d+b=w^Tx_i +b$
$h_s(x_i) = w_0x_0+w_1x_1...w_dx_d+b=w^Tx_i$
מניחים קשר לינארי בין התוכלת של x ל-y שניתן, **ריגרסיה המטרה שלה לחשב את התוכלת של y בהינתן x.**
$$E(Y_i|x_i)=w^Tx_i$$
אם בחרנו לקרב למודל לינארי והשגיאה יצאה נמוכה כלומר הנתונים באמת מתנהגים בצורה לינארית, אם לא אז השגיאה תהייה גבוהה
![[Pasted image 20241217073543.png]]

**בריגרסיה לוגיסטית** אנחנו לא רוצים את התוכלת של y בהינתן x, אלא אנחנו רוצים את ההסתברות ש-y יהיה שווה ל-0 או 1 בהתאם ללייבל
אפשר להשתמש בריגרסיה לינארית לריגרסיה לוגיסטית אם ממירים את הערך ל0 או 1.
משתמשים בפונקציית סיגמואיד בשביל זה, "לתרגם" את הערך המספרי הלינארי לערכים של 0 או 1.
