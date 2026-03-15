"""MP-Declare constraint mining via RuM (MINERful + MpEnhancer)."""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import jpype

from ..declare import DeclareConstraint, DeclareTemplate

RUM_JAR = Path(__file__).parent / "rum-0.7.2.jar"


def _ensure_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(f"-Djava.class.path={RUM_JAR}", "-Djava.awt.headless=true", convertStrings=True)
    __import__('jpype.imports')  # enables 'from java.x import Y' syntax


@dataclass
class MPDeclareConstraint:
    template: str
    activation: str
    target: str
    support: float
    data_condition: str = None

    def __str__(self):
        s = f"{self.template}[{self.activation}" + (f", {self.target}]" if self.target else "]")
        s += f" (support={self.support:.1%})"
        return s + (f" | {self.data_condition}" if self.data_condition else "")


def discover_mpdeclare(xes_path, min_support=0.9, data_conditions="ACTIVATIONS", verbose=True):
    """Discover MP-Declare constraints with data conditions from XES log."""
    _ensure_jvm()
    from tqdm.auto import tqdm

    from java.io import File
    from java.util import ArrayList
    from minerful import MinerFulMinerStarter
    from minerful.logparser import XesLogParser
    from minerful.logparser.LogEventClassifier import ClassificationType
    from minerful.miner.params import MinerFulCmdParameters
    from minerful.postprocessing.params import PostProcessingCmdParameters
    from task.discovery.mp_enhancer import MpEnhancer
    from controller.discovery import DataConditionType
    from controller.discovery.data import DiscoveredConstraint, DiscoveredActivity
    from util import ConstraintTemplate

    # Step 1: Mine with MINERful
    with tqdm(total=3, desc="MP-Declare discovery", disable=not verbose) as pbar:
        pbar.set_postfix_str("MINERful mining...")
        log_parser = XesLogParser(File(str(xes_path)), ClassificationType.NAME)
        post_params = PostProcessingCmdParameters()
        post_params.supportThreshold = float(min_support)
        post_params.confidenceThreshold = 0.0
        post_params.interestFactorThreshold = 0.0
        process_model = MinerFulMinerStarter().mine(log_parser, MinerFulCmdParameters(), post_params, log_parser.getTaskCharArchive())
        pbar.update(1)

        # Step 2: Convert to RuM format
        pbar.set_postfix_str("Converting constraints...")
        all_constraints = list(process_model.getAllConstraints())
        discovered = ArrayList()
        for c in tqdm(all_constraints, desc="Converting to RuM format", leave=False, disable=not verbose):
            params = c.getParameters()
            act = params[0].getTaskChar(0).getName() if len(params) > 0 and params[0].size() > 0 else None
            tgt = params[1].getTaskChar(0).getName() if len(params) > 1 and params[1].size() > 0 else None
            try:
                dc = DiscoveredConstraint(
                    ConstraintTemplate.getByTemplateName(str(c.getTemplateName())),
                    DiscoveredActivity(act, float(c.getSupport())) if act else None,
                    DiscoveredActivity(tgt, float(c.getSupport())) if tgt else None
                )
                dc.setConstraintSupport(float(c.getSupport()))
                discovered.add(dc)
            except:
                pass
        pbar.update(1)

        # Step 3: Apply MpEnhancer for data conditions (binary constraints only).
        _UNSUPPORTED_TEMPLATES = {"Succession", "Alternate Succession", "Chain Succession"}

        if data_conditions != "NONE":
            pbar.set_postfix_str("MpEnhancer data conditions...")
            binary = ArrayList()
            excluded_templates = set()
            for i in range(discovered.size()):
                dc = discovered.get(i)
                if dc.getActivationActivity() and dc.getTargetActivity():
                    tpl_name = str(dc.getTemplate())
                    if tpl_name in _UNSUPPORTED_TEMPLATES:
                        excluded_templates.add(tpl_name)
                        continue
                    binary.add(dc)
            if excluded_templates:
                print(f"[discover_mpdeclare] Skipping {len(excluded_templates)} template type(s) "
                      f"unsupported by MpEnhancer data conditions: {excluded_templates}")
            if binary.size() > 0:
                xlog = jpype.JClass('org.deckfour.xes.in.XesXmlParser')().parse(File(str(xes_path))).get(0)
                enhancer = MpEnhancer()
                enhancer.setConditionType(DataConditionType.valueOf(data_conditions))
                enhancer.setMinSupport(float(min_support))
                try:
                    enhancer.performMPDiscovery(xlog, binary, False, False)
                except Exception as e:
                    print(f"[discover_mpdeclare] MpEnhancer failed: {e}. Returning constraints without data conditions.")
        pbar.update(1)
        pbar.set_postfix_str("done")

    # Extract results — keep all data conditions found by MpEnhancer.
    # MpEnhancer updates constraint support to reflect the constraint WITH
    # the data condition (e.g. support=1.0 means "with this condition, the
    # constraint always holds" — the most interesting case).
    results = []
    for i in tqdm(range(discovered.size()), desc="Extracting results", leave=False, disable=not verbose):
        dc = discovered.get(i)
        support = float(dc.getConstraintSupport())
        data_cond = None
        if dc.getDataCondition():
            data_cond = dc.getDataCondition().toDeclareString().replace('&#8743;', '∧').replace('&#61;', '=')
        results.append(MPDeclareConstraint(
            template=str(dc.getTemplate()),
            activation=str(dc.getActivationActivity()) if dc.getActivationActivity() else None,
            target=str(dc.getTargetActivity()) if dc.getTargetActivity() else None,
            support=support,
            data_condition=data_cond
        ))
    return results


# =============================================================================
# RuM → DeclareConstraint conversion
# =============================================================================

# Mapping from RuM template names to (DeclareTemplate, parameter) tuples.
# parameter is only used for unary existence/absence/exactly templates.
# None value means the template is unsupported and will be skipped.
RUM_TO_DECLARE_TEMPLATE = {
    # Unary existence
    "Init":       (DeclareTemplate.INIT, None),
    "End":        (DeclareTemplate.LAST, None),
    "Existence":  (DeclareTemplate.EXISTENCE, 1),
    "Existence2": (DeclareTemplate.EXISTENCE, 2),
    "Existence3": (DeclareTemplate.EXISTENCE, 3),
    "Absence":    (DeclareTemplate.ABSENCE, 1),
    "Absence2":   (DeclareTemplate.ABSENCE, 2),
    "Absence3":   (DeclareTemplate.ABSENCE, 3),
    "Exactly1":   (DeclareTemplate.EXACTLY, 1),
    "Exactly2":   (DeclareTemplate.EXACTLY, 2),

    # Binary
    "Co-Existence":         (DeclareTemplate.CO_EXISTENCE, None),
    "Response":             (DeclareTemplate.RESPONSE, None),
    "Precedence":           (DeclareTemplate.PRECEDENCE, None),
    "Succession":           (DeclareTemplate.SUCCESSION, None),
    "Not Succession":       (DeclareTemplate.NOT_SUCCESSION, None),
    "Alternate Response":   (DeclareTemplate.ALTERNATE_RESPONSE, None),
    "Alternate Precedence": (DeclareTemplate.ALTERNATE_PRECEDENCE, None),
    "Alternate Succession": (DeclareTemplate.ALTERNATE_SUCCESSION, None),
    "Chain Response":       (DeclareTemplate.CHAIN_RESPONSE, None),
    "Chain Precedence":     (DeclareTemplate.CHAIN_PRECEDENCE, None),
    "Chain Succession":     (DeclareTemplate.CHAIN_SUCCESSION, None),

    # Unsupported — skipped during conversion
    "Choice":               None,
    "Exclusive Choice":     None,
    "Responded Existence":  None,
    "Not Chain Succession": None,
    "Not Co-Existence":     None,
    "Not Chain Precedence": None,
    "Not Chain Response":   None,
    "Not Precedence":       None,
    "Not Response":         None,
}


def _extract_activity_name(rum_str: str) -> str:
    """Extract clean activity name from RuM DiscoveredActivity string.

    RuM's DiscoveredActivity.toString() returns 'Activity: "name" (supp=X.X)'.
    This extracts just 'name'. If the string is already a clean name, returns it as-is.
    """
    if rum_str.startswith('Activity: "'):
        return rum_str.split('"')[1]
    return rum_str


def mpdeclare_to_declare_constraints(
    mpdeclare_constraints: List[MPDeclareConstraint],
    activity_name_to_idx: Dict[str, int],
) -> Tuple[Set[DeclareConstraint], Dict[DeclareConstraint, str]]:
    """Convert RuM MPDeclareConstraint results to DeclareConstraint format.

    Args:
        mpdeclare_constraints: List of MPDeclareConstraint from discover_mpdeclare().
        activity_name_to_idx: Mapping from activity name string to integer index.

    Returns:
        (constraints_set, data_conditions_dict) where:
        - constraints_set: Set of DeclareConstraint instances
        - data_conditions_dict: Dict mapping DeclareConstraint → data condition string
          (only for constraints that have data conditions)
    """
    constraints = set()
    data_conditions = {}
    skipped_templates = set()
    skipped_activities = set()

    for mc in mpdeclare_constraints:
        # Look up template mapping
        mapping = RUM_TO_DECLARE_TEMPLATE.get(mc.template)
        if mapping is None:
            skipped_templates.add(mc.template)
            continue

        declare_template, parameter = mapping

        # Extract clean activity names (RuM stores them as 'Activity: "name" (supp=X.X)')
        act_name = _extract_activity_name(mc.activation) if mc.activation else None
        tgt_name = _extract_activity_name(mc.target) if mc.target else None

        # Resolve activity indices
        if act_name is not None and act_name not in activity_name_to_idx:
            skipped_activities.add(act_name)
            continue
        if tgt_name is not None and tgt_name not in activity_name_to_idx:
            skipped_activities.add(tgt_name)
            continue

        # Build activity tuple
        if declare_template.is_unary():
            act_idx = activity_name_to_idx[act_name]
            dc = DeclareConstraint(
                template=declare_template,
                activities=(act_idx,),
                parameter=parameter,
            )
        else:
            act_idx = activity_name_to_idx[act_name]
            tgt_idx = activity_name_to_idx[tgt_name]
            dc = DeclareConstraint(
                template=declare_template,
                activities=(act_idx, tgt_idx),
                parameter=None,
            )

        constraints.add(dc)

        # Store data condition if present
        if mc.data_condition:
            data_conditions[dc] = mc.data_condition

    if skipped_templates:
        print(f"[mpdeclare_to_declare] Skipped {len(skipped_templates)} unsupported template(s): {skipped_templates}")
    if skipped_activities:
        print(f"[mpdeclare_to_declare] Skipped {len(skipped_activities)} unknown activity(ies): {skipped_activities}")

    return constraints, data_conditions
