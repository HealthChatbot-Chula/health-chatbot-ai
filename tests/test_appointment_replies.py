import unittest
from datetime import datetime

from agent.appointment_inputs import normalize_appointment_time
from agent.appointment_prompts import appointment_prompt
from agent.appointment_replies import (
    BANGKOK, confirmation_decision, confirmation_text, date_label, error_reply,
    local_slot, read_reply, reply_text, strip_foreign, suggestion_choices, write_reply,
)


FRIDAY = {"date": "2026-11-20", "weekday": "วันศุกร์", "time": "14:30"}
WEDNESDAY = {"date": "2026-11-18", "weekday": "วันพุธ", "time": "10:00"}
MOVE = {"action": "reschedule", "args": {}, "from": WEDNESDAY, "to": FRIDAY}


class AppointmentReplyTests(unittest.TestCase):
    def test_removes_foreign_character_from_screenshot(self):
        self.assertEqual(strip_foreign("ให้ช่วยดูวัน이ว่างอื่นไหมครับ"), "ให้ช่วยดูวันว่างอื่นไหมครับ")

    def test_reads_text_blocks(self):
        self.assertEqual(reply_text([{"type": "text", "text": "เวลา 12:30 น."}]), "เวลา 12:30 น.")

    def test_normalizes_dot_time_without_rounding(self):
        self.assertEqual(normalize_appointment_time("14.30"), "14:30")
        self.assertEqual(normalize_appointment_time("8:00"), "08:00")
        self.assertEqual(normalize_appointment_time("12.11"), "12:11")
        self.assertIsNone(normalize_appointment_time("14.3"))
        self.assertIsNone(normalize_appointment_time("24:00"))

    def test_calendar_instants_are_shown_in_bangkok_time(self):
        self.assertEqual(local_slot("2026-11-30T07:30:00Z"), ("2026-11-30", "14:30"))
        self.assertEqual(local_slot("2026-11-30T18:00:00Z"), ("2026-12-01", "01:00"))
        self.assertIsNone(local_slot("2026-11-30T07:30:00"))
        self.assertEqual(date_label("2026-11-20"), "วันศุกร์ที่ 20 พฤศจิกายน 2026")

    def test_short_answers_confirm_even_with_casual_endings(self):
        for text in ["ยืนยัน", "ยืนยันจร้า", "ใช่", "ใช่ครับ", "ใช่จ้าา", "ยืนยันค่ะ", "ตกลง", "OK", "ได้เลยครับ", "โอเคค่า"]:
            self.assertEqual(confirmation_decision(text), "yes", text)
        for text in ["ไม่", "ไม่ยืนยัน", "ไม่ใช่ครับ", "ไม่เอาแล้ว", "ยกเลิก", "ยังก่อนนะ"]:
            self.assertEqual(confirmation_decision(text), "no", text)

    def test_answers_with_new_details_or_doubt_are_not_a_decision(self):
        for text in ["ขอเป็น 15:00 แทน", "ใช่ แต่ขอเปลี่ยนเป็นวันจันทร์", "นัดหมายวันที่ 20/11", "ไม่รู้", "ไม่แน่ใจ",
                     "ยืนยันได้ไหม", "ใช่วันที่ 20 หรือเปล่า", "ค่าน้ำตาล 110 สูงไหม", ""]:
            self.assertIsNone(confirmation_decision(text), text)

    def test_cancel_word_follows_what_is_being_confirmed(self):
        # Answering "ยกเลิก" to "cancel this appointment?" means go ahead...
        self.assertEqual(confirmation_decision("ยกเลิกเลยครับ", "cancel"), "yes")
        self.assertEqual(confirmation_decision("ไม่ยกเลิกแล้ว", "cancel"), "no")
        # ...but to "book this slot?" it means don't.
        self.assertEqual(confirmation_decision("ยกเลิก", "create"), "no")

    def test_confirmation_states_both_old_and_new_slot(self):
        text = confirmation_text([MOVE])
        self.assertIn("วันพุธที่ 18 พฤศจิกายน 2026 เวลา 10:00", text)
        self.assertIn("วันศุกร์ที่ 20 พฤศจิกายน 2026 เวลา 14:30", text)
        self.assertNotIn("เรียบร้อย", text)

    def test_confirmation_lists_every_operation(self):
        nine = {"action": "cancel", "args": {}, "from": {"date": "2026-11-18", "time": "09:00"}}
        noon = {"action": "cancel", "args": {}, "from": {"date": "2026-11-18", "time": "12:00"}}
        text = confirmation_text([nine, noon])
        self.assertIn("2 รายการ", text)
        self.assertIn("09:00", text)
        self.assertIn("12:00", text)

    def test_suggested_times_become_self_contained_buttons(self):
        times = [f"{h:02d}:{m}" for h in range(8, 16) for m in ("00", "30")]
        moved = suggestion_choices({"suggestion": {"date": "2026-11-20", "times": times},
                                    "requested": {"date": "2026-11-20", "time": "14:14"},
                                    "retry": {"action": "reschedule", "from": WEDNESDAY}})
        self.assertEqual([item["label"] for item in moved], ["13:30", "14:00", "14:30", "15:00"])
        self.assertEqual(moved[1]["value"], "ขอเลื่อนนัดวันที่ 18/11/2026 เวลา 10:00 เป็นวันที่ 20/11/2026 เวลา 14:00")
        booked = suggestion_choices({"suggestion": {"date": "2026-11-23", "times": ["09:00"]},
                                     "requested": {"date": "2026-11-21", "time": "09:00"}, "retry": {"action": "create"}})
        self.assertEqual(booked, [{"label": "23/11 09:00", "value": "ขอจองนัดวันที่ 23/11/2026 เวลา 09:00"}])
        self.assertIsNone(suggestion_choices({"error": "x", "code": "ambiguous"}))

    def test_success_is_claimed_only_from_a_confirmed_result(self):
        done = write_reply(MOVE, {"status": "confirmed", "appointment": {"start": "2026-11-20T07:30:00Z"}})
        self.assertIn("เลื่อนนัดหมายเรียบร้อย", done)
        self.assertIn("14:30", done)
        for result in [{"error": "offline", "code": "unknown_outcome"}, {"error": "x", "status": 503}, {}]:
            reply = write_reply(MOVE, result)
            self.assertIn("ยังยืนยัน", reply)
            self.assertNotIn("เรียบร้อย", reply)

    def test_refusal_gives_backend_reason_and_nearest_free_times(self):
        reply = error_reply({
            "error": "นัดหมายเป็นช่วงละ 30 นาที เริ่มได้เฉพาะนาทีที่ :00 หรือ :30", "status": 400, "code": "off_grid",
            "requested": {"date": "2026-11-16", "time": "08:15"},
            "suggestion": {"date": "2026-11-16", "times": [f"{h:02d}:{m}" for h in range(8, 16) for m in ("00", "30")]},
        })
        self.assertIn("30 นาที", reply)
        self.assertIn("08:00, 08:30", reply)
        self.assertNotIn("15:30", reply)
        closed = error_reply({"error": "วันดังกล่าวเป็นวันหยุด เปิดให้นัดเฉพาะวันจันทร์–ศุกร์", "status": 400,
                              "code": "closed_day", "suggestion": {"date": "2026-11-23", "times": ["09:00"]}})
        self.assertIn("วันหยุด", closed)
        self.assertIn("วันจันทร์ที่ 23 พฤศจิกายน 2026", closed)

    def test_read_fallback_never_claims_a_booking(self):
        self.assertNotIn("บันทึก", read_reply({"appointments": [WEDNESDAY]}))
        self.assertIn("วันหยุด", read_reply({"date": "2026-11-21", "free_times": [], "reason": "closed_day"}))

    def test_prompt_lists_real_weekdays_instead_of_leaving_date_maths_to_the_model(self):
        prompt = appointment_prompt(now=datetime(2026, 11, 15, 18, 0, tzinfo=BANGKOK))
        self.assertIn("2026-11-15 วันอาทิตย์ (วันนี้) - closed", prompt)
        self.assertIn("2026-11-20 วันศุกร์", prompt)
        self.assertLess(len(prompt.splitlines()), 32)
        self.assertIn("date just discussed", prompt)


if __name__ == "__main__":
    unittest.main()
