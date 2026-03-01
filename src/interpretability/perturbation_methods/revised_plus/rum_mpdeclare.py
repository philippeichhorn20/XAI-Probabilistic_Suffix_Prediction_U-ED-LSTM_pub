"""MP-Declare constraint mining via RuM (MINERful + MpEnhancer)."""

from pathlib import Path
from dataclasses import dataclass
import jpype

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


def discover_mpdeclare(xes_path, min_support=0.9, data_conditions="ACTIVATIONS", data_condition_threshold=0.9):
    """Discover MP-Declare constraints with data conditions from XES log.

    Data conditions are only kept for constraints with base support < data_condition_threshold.
    High-support constraints don't need data conditions as they hold universally.
    """
    _ensure_jvm()

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

    # Mine with MINERful
    log_parser = XesLogParser(File(str(xes_path)), ClassificationType.NAME)
    post_params = PostProcessingCmdParameters()
    post_params.supportThreshold = float(min_support)
    post_params.confidenceThreshold = 0.0
    post_params.interestFactorThreshold = 0.0
    process_model = MinerFulMinerStarter().mine(log_parser, MinerFulCmdParameters(), post_params, log_parser.getTaskCharArchive())

    # Convert to RuM format
    discovered = ArrayList()
    for c in process_model.getAllConstraints():
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

    # Apply MpEnhancer for data conditions (binary constraints only)
    if data_conditions != "NONE":
        binary = ArrayList()
        for i in range(discovered.size()):
            dc = discovered.get(i)
            if dc.getActivationActivity() and dc.getTargetActivity():
                binary.add(dc)
        if binary.size() > 0:
            xlog = jpype.JClass('org.deckfour.xes.in.XesXmlParser')().parse(File(str(xes_path))).get(0)
            enhancer = MpEnhancer()
            enhancer.setConditionType(DataConditionType.valueOf(data_conditions))
            enhancer.setMinSupport(float(min_support))
            enhancer.performMPDiscovery(xlog, binary, False, False)

    # Extract results (filter out data conditions for high-support constraints)
    results = []
    for i in range(discovered.size()):
        dc = discovered.get(i)
        support = float(dc.getConstraintSupport())
        # Only keep data condition if support < threshold (otherwise constraint holds universally)
        data_cond = None
        if dc.getDataCondition() and support < data_condition_threshold:
            data_cond = dc.getDataCondition().toDeclareString().replace('&#8743;', '∧').replace('&#61;', '=')
        results.append(MPDeclareConstraint(
            template=str(dc.getTemplate()),
            activation=str(dc.getActivationActivity()) if dc.getActivationActivity() else None,
            target=str(dc.getTargetActivity()) if dc.getTargetActivity() else None,
            support=support,
            data_condition=data_cond
        ))
    return results
