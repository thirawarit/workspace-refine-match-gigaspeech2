# USER REQUIREMENTS

1. Use model from HF repo: `typhoon-ai/typhoon-asr-streaming-nemotron-0.6b`, 
2. Accept input in TSV or JSONL format and output in the same format as the input when inferencing using an ASR model. For example, the output is `predicted-AbC.tsv` if the input is `AbC.tsv`.
3. After the interim process, combine the input file and the prediction output into a new file. The new file includes columns such as `segment_id`, `orig_text`, and `pred_text`.

## Sample Content

```tsv
100-100000-0	ท่านผู้ชมครับเรื่องของ COVID-19 วันนี้ไทยพบผู้ป่วยเพิ่มหนึ่งคนนะครับ
100-100000-1	เป็นผู้หญิงอายุยี่สิบสองอาชีพดูแลนักท่องเที่ยว
100-100000-2	สัมผัสกับกลุ่มผู้ที่มีความเสี่ยงสูง
... ...
148-148801-10   ดำเนินการจัดการเคลียร์พื้นที่ดังกล่าว
148-148801-11	ให้ทันภายใน24พฤศจิกายนนี้
```