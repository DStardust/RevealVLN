"""Pure CPU helper tests; do not start monitor, nvidia-smi, or signal processes."""
import unittest
if __package__:
    from . import guard
else:
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("supplemental_guard_cpu", Path(__file__).resolve().with_name("guard.py"))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
UUID, SUPERVISOR, WORKER = guard.UUID, guard.SUPERVISOR, guard.WORKER
EXPECTED_START, EXPECTED_CMD, EXPECTED_CWD = guard.EXPECTED_START, guard.EXPECTED_CMD, guard.EXPECTED_CWD
mib, parse_xml, assess, parse_stat, validate_identity = guard.mib, guard.parse_xml, guard.assess, guard.parse_stat, guard.validate_identity


def xml(total=1000, rows=((WORKER, 500, "G"), (22, 100, "C")), uuid=UUID):
    processes = "".join(f"<process_info><pid>{p}</pid><type>{kind}</type><used_memory>{n} MiB</used_memory></process_info>" for p,n,kind in rows)
    return f"<nvidia_smi_log><gpu><uuid>{uuid}</uuid><fb_memory_usage><used>{total} MiB</used></fb_memory_usage><processes>{processes}</processes></gpu></nvidia_smi_log>"


class SupplementalGuardTests(unittest.TestCase):
    def test_graphics_renderer_included(self):
        data = parse_xml(xml())
        self.assertEqual(data["processes"][WORKER]["type"], "G")
        report = assess(data, {WORKER})
        self.assertTrue(report["pass"])
        self.assertEqual(report["own_conservative_upper_mib"], 900)

    def test_uuid_rejected(self):
        with self.assertRaises(ValueError):
            parse_xml(xml(uuid="OTHER"))

    def test_unknown_memory_rejected(self):
        for value in ("N/A", "12 GiB", "-1 MiB", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mib(value)

    def test_duplicate_pid_rejected(self):
        with self.assertRaises(ValueError):
            parse_xml(xml(rows=((22, 100, "C"), (22, 100, "G"))))

    def test_each_external_bound(self):
        self.assertTrue(assess(parse_xml(xml(rows=((22,768,"C"),))), set())["pass"])
        self.assertFalse(assess(parse_xml(xml(rows=((22,769,"C"),))), set())["pass"])

    def test_external_total_bound(self):
        rows = ((22,700,"C"),(23,700,"G"),(24,649,"G"))
        report = assess(parse_xml(xml(total=3000,rows=rows)),set())
        self.assertIn("EXTERNAL_TOTAL_EXCEEDS_2048_MIB", report["reasons"])

    def test_own_upper_strict_limit(self):
        for total, passed in ((4195, True), (4196, False)):
            self.assertEqual(assess(parse_xml(xml(total=total)),{WORKER})["pass"], passed)

    def test_accounting_inconsistent_fail_closed(self):
        self.assertFalse(assess(parse_xml(xml(total=300)),{WORKER})["pass"])

    def test_proc_stat_start_ticks_not_pid_only(self):
        fields = ["S", "42"] + ["0"] * 17 + ["999"]
        self.assertEqual(parse_stat("123 (name with ) chars) " + " ".join(fields)),
                         {"pid":123,"ppid":42,"start_ticks":999})

    def identity(self, pid=WORKER):
        return {"pid":pid,"ppid":SUPERVISOR,"start_ticks":EXPECTED_START[pid],
                "cmd":EXPECTED_CMD[pid][:],"cwd":str(EXPECTED_CWD[pid])}

    def test_exact_identity(self):
        self.assertTrue(validate_identity(self.identity(), WORKER))
        for key,value in (("start_ticks",1),("cmd",["unrelated"]),("cwd","/tmp"),("ppid",1)):
            identity=self.identity();identity[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_identity(identity,WORKER)

    def test_wallclock_rollback_does_not_extend_budget(self):
        boot = EXPECTED_START[SUPERVISOR] / 100 + 3001
        self.assertLess(guard.remaining_budget(10000, 9999, boot, 100), 0)


if __name__ == "__main__":
    unittest.main()
