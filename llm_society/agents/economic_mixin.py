"""
Economic Agent Mixin - Trading, banking, and resource management
"""

import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from llm_society.economics.market_system import ResourceType, TradeOrderType
    from llm_society.economics.banking_system import (
        AccountType,
        LoanType,
        LoanStatus,
        TransactionType,
    )

logger = logging.getLogger(__name__)


class EconomicMixin:
    """Mixin providing economic capabilities to agents"""

    resources: Dict[str, int]
    inventory: List[str]
    credit_score: float
    total_debt: float
    monthly_income: float
    unique_id: str
    model: Any
    energy: float

    def _init_economic(self):
        """Initialize economic attributes"""
        self.resources = {
            "food": 10,
            "materials": 5,
            "tools": 1,
            "currency": 500,
            "energy_item": 5,
        }
        self.inventory = []
        self.credit_score = 700.0
        self.total_debt = 0.0
        self.monthly_income = 0.0

    def _update_resources(self):
        """Update energy and resource decay"""
        # Gradual energy decay
        self.energy = max(0.0, self.energy - 0.005)

        # Resource consumption
        if self.step_count % 20 == 0:  # Every 20 steps
            self.resources["food"] = max(0, self.resources["food"] - 1)
            if self.resources["food"] == 0:
                self.energy = max(0.0, self.energy - 0.1)  # Hunger effect

    async def _execute_gather(self):
        """Execute resource gathering"""
        resource_type = random.choice(["food", "materials"])
        amount = random.randint(1, 3)
        self.resources[resource_type] += amount
        self.energy -= 0.05

        await self._add_memory(f"Gathered {amount} {resource_type}", importance=0.3)

    async def _execute_market_trade(
        self, order_type_str: str, resource_str: str, quantity: float, price: float
    ):
        """Executes a market trade by submitting an order to the MarketSystem."""
        from llm_society.economics.market_system import ResourceType, TradeOrderType

        try:
            order_type = TradeOrderType(order_type_str.lower())
            resource_type = ResourceType(resource_str.lower())
        except ValueError as e:
            logger.warning(
                f"Agent {self.unique_id} provided invalid trade parameters: {order_type_str}, {resource_str}. Error: {e}"
            )
            await self._add_memory(
                f"Failed market trade due to invalid parameters: {order_type_str} {resource_str}",
                importance=0.5,
            )
            return

        logger.info(
            f"Agent {self.unique_id} attempting market trade: {order_type.value} {quantity} {resource_type.value} at {price}"
        )

        order_id = self.model.market_system.submit_order(
            agent_id=self.unique_id,
            resource_type=resource_type,
            order_type=order_type,
            quantity=quantity,
            price_per_unit=price,
        )

        if order_id:
            await self._add_memory(
                f"Submitted market order {order_id}: {order_type.value} {quantity} {resource_type.value} at {price}",
                importance=0.7,
            )
            logger.info(f"Agent {self.unique_id} submitted order {order_id} to market.")
        else:
            await self._add_memory(
                f"Failed to submit market order: {order_type.value} {quantity} {resource_type.value}",
                importance=0.5,
            )
            logger.warning(f"Agent {self.unique_id} failed to submit market order.")

        self.energy -= 0.05

    async def _execute_market_research(self, target: str):
        """Executes market research by querying the MarketSystem."""
        from llm_society.economics.market_system import ResourceType

        logger.info(f"Agent {self.unique_id} performing market research for: {target}")
        market_info_result = "No market information obtained."
        self.last_market_research_result = None

        if not hasattr(self.model, "market_system"):
            logger.warning(
                f"MarketSystem not found on model for agent {self.unique_id}."
            )
            await self._add_memory(
                f"Attempted market research for {target}, but market system is unavailable.",
                importance=0.3,
            )
            return

        try:
            if target == "all":
                if hasattr(self.model.market_system, "get_general_market_overview"):
                    market_info_result = (
                        self.model.market_system.get_general_market_overview()
                    )
                    if not market_info_result:
                        market_info_result = (
                            "General market overview is currently empty or unavailable."
                        )
                else:
                    market_info_result = (
                        "Function to get general market overview is not available."
                    )
            else:
                if hasattr(
                    self.model.market_system, "get_detailed_resource_market_info"
                ):
                    try:
                        resource_enum = ResourceType(target.lower())
                        market_info_result = (
                            self.model.market_system.get_detailed_resource_market_info(
                                resource_enum
                            )
                        )
                        if not market_info_result:
                            market_info_result = f"Detailed market info for {target} is currently empty or unavailable."
                    except ValueError:
                        market_info_result = (
                            f"Unknown resource '{target}' for market research."
                        )
                else:
                    market_info_result = f"Function to get detailed market info for {target} is not available."

            await self._add_memory(
                f"Market research for '{target}': {market_info_result[:200]}...",
                importance=0.6,
            )
            self.last_market_research_result = market_info_result
            logger.info(
                f"Agent {self.unique_id} market research result for '{target}': {market_info_result}"
            )

        except Exception as e:
            logger.error(
                f"Error during market research for agent {self.unique_id} (target: {target}): {e}",
                exc_info=True,
            )
            await self._add_memory(
                f"Error performing market research for '{target}'.", importance=0.4
            )
            self.last_market_research_result = (
                f"Error during market research for {target}."
            )

        self.energy -= 0.01

    async def _execute_banking_action(
        self,
        sub_action_str: str,
        amount: Optional[float],
        loan_details_str: Optional[str] = None,
        loan_id_str: Optional[str] = None,
    ):
        """Executes a banking action by calling the BankingSystem."""
        from llm_society.economics.banking_system import (
            AccountType,
            LoanType,
            TransactionType,
        )

        agent_id = self.unique_id
        if not hasattr(self.model, "banking_system") or not self.model.banking_system:
            logger.warning(f"BankingSystem not found on model for agent {agent_id}.")
            await self._add_memory(
                "Tried banking action, but system is unavailable.", importance=0.3
            )
            return
        banking_system = self.model.banking_system
        current_sim_step = self.model.current_step

        primary_account_id = None
        if (
            agent_id in banking_system.agent_accounts
            and banking_system.agent_accounts[agent_id]
        ):
            primary_account_id = banking_system.agent_accounts[agent_id][0]

        # Auto-create account for deposit/withdraw if none exists
        if sub_action_str in ["deposit", "withdraw"] and not primary_account_id:
            logger.info(
                f"Agent {agent_id} has no account for {sub_action_str}. Attempting to create one."
            )
            new_account = await banking_system.create_account(
                agent_id=agent_id,
                account_type=AccountType.CHECKING,
                initial_deposit=0.0,
                current_step=current_sim_step,
            )
            if new_account:
                primary_account_id = new_account.account_id
                await self._add_memory(
                    f"Auto-opened checking account {primary_account_id} for {sub_action_str}.",
                    importance=0.6,
                )
                logger.info(
                    f"Agent {agent_id} auto-created account {primary_account_id}."
                )
            else:
                logger.warning(
                    f"Failed to auto-create bank account for {agent_id}."
                )
                await self._add_memory(
                    f"Banking action ({sub_action_str}) failed: could not ensure bank account.",
                    importance=0.5,
                )
                self.energy -= 0.02
                return

        try:
            if sub_action_str == "deposit":
                await self._handle_deposit(
                    banking_system, primary_account_id, amount, current_sim_step
                )
            elif sub_action_str == "withdraw":
                await self._handle_withdraw(
                    banking_system, primary_account_id, amount, current_sim_step
                )
            elif sub_action_str == "apply_loan":
                await self._handle_apply_loan(
                    banking_system, amount, loan_details_str, current_sim_step
                )
            elif sub_action_str == "pay_loan":
                await self._handle_pay_loan(
                    banking_system, loan_id_str, amount, current_sim_step
                )
            else:
                logger.warning(
                    f"{agent_id} unknown banking sub-action: {sub_action_str}"
                )
                await self._add_memory(f"Unknown banking action: {sub_action_str}", 0.3)

        except Exception as e:
            logger.error(
                f"Error banking action for {agent_id}: {sub_action_str}. Error: {e}",
                exc_info=True,
            )
            await self._add_memory(f"Error banking action '{sub_action_str}'.", 0.5)

        self.energy -= 0.02

    async def _handle_deposit(self, banking_system, account_id, amount, current_step):
        """Handle deposit banking action"""
        from llm_society.economics.banking_system import TransactionType

        if amount is None or amount <= 0:
            logger.warning(f"Deposit for {self.unique_id} invalid amount: {amount}.")
            await self._add_memory("Tried deposit with invalid amount.", 0.4)
            return
        if not account_id:
            logger.warning(f"{self.unique_id} no account for deposit.")
            await self._add_memory("Tried deposit, no account.", 0.4)
            return
        if self.resources.get("currency", 0) >= amount:
            success = await banking_system.process_transaction(
                account_id=account_id,
                transaction_type=TransactionType.DEPOSIT,
                amount=amount,
                description="Agent deposit",
                current_step=current_step,
            )
            if success:
                self.resources["currency"] -= amount
                await self._add_memory(f"Deposited {amount} to {account_id}.", 0.7)
                logger.info(f"{self.unique_id} deposited {amount}.")
            else:
                await self._add_memory(f"Failed deposit {amount} to {account_id}.", 0.5)
                logger.warning(f"{self.unique_id} failed deposit {amount}.")
        else:
            await self._add_memory(f"Tried deposit {amount}, insufficient cash.", 0.5)
            logger.warning(f"{self.unique_id} insufficient cash for deposit.")

    async def _handle_withdraw(self, banking_system, account_id, amount, current_step):
        """Handle withdraw banking action"""
        from llm_society.economics.banking_system import TransactionType

        if amount is None or amount <= 0:
            logger.warning(f"Withdraw for {self.unique_id} invalid amount: {amount}.")
            await self._add_memory("Tried withdraw with invalid amount.", 0.4)
            return
        if not account_id:
            logger.warning(f"{self.unique_id} no account for withdraw.")
            await self._add_memory("Tried withdraw, no account.", 0.4)
            return
        account = banking_system.accounts.get(account_id)
        if account and account.balance >= amount:
            success = await banking_system.process_transaction(
                account_id=account_id,
                transaction_type=TransactionType.WITHDRAWAL,
                amount=amount,
                description="Agent withdrawal",
                current_step=current_step,
            )
            if success:
                self.resources["currency"] = self.resources.get("currency", 0) + amount
                await self._add_memory(f"Withdrew {amount} from {account_id}.", 0.7)
                logger.info(f"{self.unique_id} withdrew {amount}.")
            else:
                await self._add_memory(
                    f"Failed withdraw {amount} from {account_id}.", 0.5
                )
                logger.warning(f"{self.unique_id} failed withdraw {amount}.")
        else:
            await self._add_memory(
                f"Tried withdraw {amount} from {account_id}, insufficient balance.", 0.5
            )
            logger.warning(f"{self.unique_id} insufficient balance for withdraw.")

    async def _handle_apply_loan(
        self, banking_system, amount, loan_details_str, current_step
    ):
        """Handle apply loan banking action"""
        from llm_society.economics.banking_system import LoanType

        if amount is None or amount <= 0:
            logger.warning(f"Apply_loan for {self.unique_id} invalid amount: {amount}.")
            await self._add_memory("Tried apply_loan with invalid amount.", 0.4)
            return
        loan_purpose = loan_details_str or "personal expenses"
        loan_type_enum = LoanType.PERSONAL
        if "business" in loan_purpose.lower():
            loan_type_enum = LoanType.BUSINESS
        elif "education" in loan_purpose.lower():
            loan_type_enum = LoanType.EDUCATION
        elif "house" in loan_purpose.lower():
            loan_type_enum = LoanType.MORTGAGE

        loan_app = await banking_system.apply_for_loan(
            self.unique_id,
            loan_type_enum,
            amount,
            loan_purpose,
            36,
            current_step=current_step,
        )
        if loan_app:
            await self._add_memory(
                f"Applied for {loan_type_enum.value} loan of {amount} for '{loan_purpose}'. ID: {loan_app.loan_id}",
                0.8,
            )
            logger.info(f"{self.unique_id} applied for loan {loan_app.loan_id}.")
        else:
            await self._add_memory(
                f"Failed to apply for loan of {amount} for '{loan_purpose}'.", 0.6
            )
            logger.warning(f"{self.unique_id} failed to apply for loan.")

    async def _handle_pay_loan(self, banking_system, loan_id_str, amount, current_step):
        """Handle pay loan banking action"""
        if not loan_id_str or amount is None or amount <= 0:
            logger.warning(
                f"Pay_loan for {self.unique_id} invalid params: {loan_id_str}, {amount}."
            )
            await self._add_memory("Tried pay_loan with invalid/missing params.", 0.4)
            return
        logger.info(
            f"{self.unique_id} attempting to pay {amount} for loan {loan_id_str}."
        )
        success = await banking_system.process_loan_payment(
            loan_id_str, amount, current_step=current_step
        )
        if success:
            await self._add_memory(f"Paid {amount} for loan {loan_id_str}.", 0.85)
            logger.info(f"{self.unique_id} paid {amount} for loan {loan_id_str}.")
        else:
            await self._add_memory(
                f"Failed payment of {amount} for loan {loan_id_str}. Check funds/status.",
                0.6,
            )
            logger.warning(
                f"{self.unique_id} failed to pay {amount} for loan {loan_id_str}."
            )

    async def _execute_get_banking_statement(self):
        """Retrieves and records the agent's detailed banking statement."""
        from llm_society.economics.banking_system import LoanStatus

        logger.info(f"Agent {self.unique_id} requesting banking statement.")
        self.last_banking_statement = None
        statement_str = "Banking statement unavailable or no accounts found."

        if not hasattr(self.model, "banking_system") or not hasattr(
            self.model.banking_system, "get_agent_financial_summary"
        ):
            logger.warning(
                f"BankingSystem or get_agent_financial_summary method not found for agent {self.unique_id}."
            )
            await self._add_memory(
                "Tried to get banking statement, but system/method is unavailable.",
                importance=0.3,
            )
            self.last_banking_statement = (
                "System error: Banking statement function unavailable."
            )
            self.energy -= 0.01
            return

        try:
            summary_dict = self.model.banking_system.get_agent_financial_summary(
                self.unique_id
            )
            if summary_dict and "error" not in summary_dict:
                parts = [f"Financial Summary for {self.unique_id}:"]
                parts.append(
                    f"  Total Balance: {summary_dict.get('total_balance', 0.0):.2f}"
                )
                parts.append(f"  Total Debt: {summary_dict.get('total_debt', 0.0):.2f}")
                parts.append(f"  Net Worth: {summary_dict.get('net_worth', 0.0):.2f}")
                parts.append(
                    f"  Credit Score: {summary_dict.get('credit_score', 0.0):.0f}"
                )

                account_details = summary_dict.get("account_details", [])
                if account_details:
                    parts.append(f"  Accounts ({len(account_details)}):")
                    for acc in account_details[:2]:
                        parts.append(
                            f"    - ID: {acc.get('account_id')}, Type: {acc.get('type')}, Balance: {acc.get('balance',0):.2f}"
                        )
                else:
                    parts.append("  No bank accounts found.")

                loan_details = summary_dict.get("loan_details", [])
                active_loans = [
                    loan
                    for loan in loan_details
                    if loan.get("status") == LoanStatus.ACTIVE.value
                ]
                if active_loans:
                    parts.append(f"  Active Loans ({len(active_loans)}):")
                    for loan in active_loans[:2]:
                        parts.append(
                            f"    - ID: {loan.get('loan_id')}, Type: {loan.get('type')}, Remaining: {loan.get('remaining_balance',0):.2f}, Payment: {loan.get('monthly_payment',0):.2f}"
                        )
                else:
                    parts.append("  No active loans found.")
                statement_str = "\n".join(parts)
            elif summary_dict and "error" in summary_dict:
                statement_str = f"Banking statement error: {summary_dict['error']}"

            await self._add_memory(
                f"Retrieved banking statement: {statement_str[:250]}...", importance=0.7
            )
            self.last_banking_statement = statement_str
            logger.info(f"Agent {self.unique_id} banking statement: {statement_str}")

        except Exception as e:
            logger.error(
                f"Error during get_banking_statement for agent {self.unique_id}: {e}",
                exc_info=True,
            )
            await self._add_memory(
                "Error retrieving banking statement.", importance=0.4
            )
            self.last_banking_statement = "Error retrieving banking statement."

        self.energy -= 0.01
