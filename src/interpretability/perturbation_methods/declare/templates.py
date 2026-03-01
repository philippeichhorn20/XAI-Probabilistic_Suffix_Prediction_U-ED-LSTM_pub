"""Declare constraint templates."""

from enum import Enum


class DeclareTemplate(Enum):
    """Supported Declare templates."""

    # Unary
    INIT = "init"
    LAST = "last"
    EXISTENCE = "existence"
    ABSENCE = "absence"
    EXACTLY = "exactly"

    # Binary
    CO_EXISTENCE = "co_existence"
    RESPONSE = "response"
    PRECEDENCE = "precedence"
    SUCCESSION = "succession"
    NOT_SUCCESSION = "not_succession"
    ALTERNATE_RESPONSE = "alternate_response"
    ALTERNATE_PRECEDENCE = "alternate_precedence"
    ALTERNATE_SUCCESSION = "alternate_succession"
    CHAIN_RESPONSE = "chain_response"
    CHAIN_PRECEDENCE = "chain_precedence"
    CHAIN_SUCCESSION = "chain_succession"

    def is_unary(self) -> bool:
        return self in (self.INIT, self.LAST, self.EXISTENCE, self.ABSENCE, self.EXACTLY)

    def is_binary(self) -> bool:
        return not self.is_unary()

    def requires_parameter(self) -> bool:
        return self in (self.EXISTENCE, self.ABSENCE, self.EXACTLY)

    def is_prefix_safe(self) -> bool:
        """Whether violations of this template on a prefix are definitive.

        A prefix-safe template has monotonic violations: once violated on
        a prefix, no suffix extension can fix it. This makes them reliable
        for scoring counterfactual prefixes without seeing the full trace.
        """
        return self in (
            self.INIT,              # first event is fixed
            self.ABSENCE,           # if A appeared, can't un-appear
            self.PRECEDENCE,        # if B without prior A, unfixable
            self.ALTERNATE_PRECEDENCE,  # same, with alternation
            self.CHAIN_PRECEDENCE,  # if B without immediately prior A, unfixable
            self.NOT_SUCCESSION,    # if A then B occurred, unfixable
        )
