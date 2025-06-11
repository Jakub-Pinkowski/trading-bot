from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.utils.logger import get_logger

logger = get_logger('backtesting/strategy_factory')


# TODO [MEDIUM]: All strategy types should be here at the top instead of line 41
class StrategyFactory:
    """
    A factory class for creating strategy instances.

    This class implements the Factory pattern to create different strategy instances
    based on strategy type and parameters. It provides a centralized place for
    strategy creation and parameter validation.

    Benefits:
    - Centralized strategy creation
    - Parameter validation
    - Easy addition of new strategy types
    - Consistent naming conventions
    """

    @staticmethod
    def create_strategy(strategy_type, **params):
        """
        Create a strategy instance based on strategy type and parameters.

        Args:
            strategy_type (str): The type of strategy to create
            **params: Strategy-specific parameters

        Returns:
            BaseStrategy: A strategy instance

        Raises:
            ValueError: If the strategy type is unknown or parameters are invalid
        """
        # Validate strategy type
        if strategy_type.lower() not in ['rsi', 'ema', 'macd', 'bollinger']:
            logger.error(f"Unknown strategy type: {strategy_type}")
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Create a strategy based on type
        if strategy_type.lower() == 'rsi':
            return StrategyFactory._create_rsi_strategy(**params)
        elif strategy_type.lower() == 'ema':
            return StrategyFactory._create_ema_strategy(**params)
        elif strategy_type.lower() == 'macd':
            return StrategyFactory._create_macd_strategy(**params)
        elif strategy_type.lower() == 'bollinger':
            return StrategyFactory._create_bollinger_strategy(**params)
        return None

    @staticmethod
    def _create_rsi_strategy(**params):
        """
        Create an RSI strategy instance.

        Args:
            **params: RSI strategy parameters

        Returns:
            RSIStrategy: An RSI strategy instance

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Extract parameters with defaults
        rsi_period = params.get('rsi_period', 14)
        lower = params.get('lower', 30)
        upper = params.get('upper', 70)
        rollover = params.get('rollover', False)
        trailing = params.get('trailing', None)

        # Validate parameters
        if not isinstance(rsi_period, int) or rsi_period <= 0:
            logger.error(f"Invalid RSI period: {rsi_period}")
            raise ValueError(f"RSI period must be a positive integer")

        if not isinstance(lower, (int, float)) or lower < 0 or lower > 100:
            logger.error(f"Invalid lower threshold: {lower}")
            raise ValueError(f"Lower threshold must be between 0 and 100")

        if not isinstance(upper, (int, float)) or upper < 0 or upper > 100:
            logger.error(f"Invalid upper threshold: {upper}")
            raise ValueError(f"Upper threshold must be between 0 and 100")

        if lower >= upper:
            logger.error(f"Lower threshold ({lower}) must be less than upper threshold ({upper})")
            raise ValueError(f"Lower threshold must be less than upper threshold")

        # Create and return strategy
        return RSIStrategy(
            rsi_period=rsi_period,
            lower=lower,
            upper=upper,
            rollover=rollover,
            trailing=trailing
        )

    @staticmethod
    def _create_ema_strategy(**params):
        """
        Create an EMA Crossover strategy instance.

        Args:
            **params: EMA strategy parameters

        Returns:
            EMACrossoverStrategy: An EMA Crossover strategy instance

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Extract parameters with defaults
        ema_short = params.get('ema_short', 9)
        ema_long = params.get('ema_long', 21)
        rollover = params.get('rollover', False)
        trailing = params.get('trailing', None)

        # Validate parameters
        if not isinstance(ema_short, int) or ema_short <= 0:
            logger.error(f"Invalid short EMA period: {ema_short}")
            raise ValueError(f"Short EMA period must be a positive integer")

        if not isinstance(ema_long, int) or ema_long <= 0:
            logger.error(f"Invalid long EMA period: {ema_long}")
            raise ValueError(f"Long EMA period must be a positive integer")

        if ema_short >= ema_long:
            logger.error(f"Short EMA period ({ema_short}) must be less than long EMA period ({ema_long})")
            raise ValueError(f"Short EMA period must be less than long EMA period")

        # Create and return strategy
        return EMACrossoverStrategy(
            ema_short=ema_short,
            ema_long=ema_long,
            rollover=rollover,
            trailing=trailing
        )

    @staticmethod
    def _create_macd_strategy(**params):
        """
        Create a MACD strategy instance.

        Args:
            **params: MACD strategy parameters

        Returns:
            MACDStrategy: A MACD strategy instance

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Extract parameters with defaults
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        rollover = params.get('rollover', False)
        trailing = params.get('trailing', None)

        # Validate parameters
        if not isinstance(fast_period, int) or fast_period <= 0:
            logger.error(f"Invalid fast period: {fast_period}")
            raise ValueError(f"Fast period must be a positive integer")

        if not isinstance(slow_period, int) or slow_period <= 0:
            logger.error(f"Invalid slow period: {slow_period}")
            raise ValueError(f"Slow period must be a positive integer")

        if not isinstance(signal_period, int) or signal_period <= 0:
            logger.error(f"Invalid signal period: {signal_period}")
            raise ValueError(f"Signal period must be a positive integer")

        if fast_period >= slow_period:
            logger.error(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
            raise ValueError(f"Fast period must be less than slow period")

        # Create and return strategy
        return MACDStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            rollover=rollover,
            trailing=trailing
        )

    @staticmethod
    def _create_bollinger_strategy(**params):
        """
        Create a Bollinger Bands strategy instance.

        Args:
            **params: Bollinger Bands strategy parameters

        Returns:
            BollingerBandsStrategy: A Bollinger Bands strategy instance

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Extract parameters with defaults
        period = params.get('period', 20)
        num_std = params.get('num_std', 2)
        rollover = params.get('rollover', False)
        trailing = params.get('trailing', None)

        # Validate parameters
        if not isinstance(period, int) or period <= 0:
            logger.error(f"Invalid period: {period}")
            raise ValueError(f"Period must be a positive integer")

        if not isinstance(num_std, (int, float)) or num_std <= 0:
            logger.error(f"Invalid number of standard deviations: {num_std}")
            raise ValueError(f"Number of standard deviations must be positive")

        # Create and return strategy
        return BollingerBandsStrategy(
            period=period,
            num_std=num_std,
            rollover=rollover,
            trailing=trailing
        )

    @staticmethod
    def get_strategy_name(strategy_type, **params):
        """
        Get a standardized name for a strategy with the given parameters.

        Args:
            strategy_type (str): The type of strategy
            **params: Strategy-specific parameters

        Returns:
            str: A standardized strategy name
        """
        if strategy_type.lower() == 'rsi':
            rsi_period = params.get('rsi_period', 14)
            lower = params.get('lower', 30)
            upper = params.get('upper', 70)
            rollover = params.get('rollover', False)
            trailing = params.get('trailing', None)
            return f'RSI(period={rsi_period},lower={lower},upper={upper},rollover={rollover},trailing={trailing})'

        elif strategy_type.lower() == 'ema':
            ema_short = params.get('ema_short', 9)
            ema_long = params.get('ema_long', 21)
            rollover = params.get('rollover', False)
            trailing = params.get('trailing', None)
            return f'EMA(short={ema_short},long={ema_long},rollover={rollover},trailing={trailing})'

        elif strategy_type.lower() == 'macd':
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)
            rollover = params.get('rollover', False)
            trailing = params.get('trailing', None)
            return f'MACD(fast={fast_period},slow={slow_period},signal={signal_period},rollover={rollover},trailing={trailing})'

        elif strategy_type.lower() == 'bollinger':
            period = params.get('period', 20)
            num_std = params.get('num_std', 2)
            rollover = params.get('rollover', False)
            trailing = params.get('trailing', None)
            return f'BB(period={period},std={num_std},rollover={rollover},trailing={trailing})'

        else:
            return f'Unknown({strategy_type})'
