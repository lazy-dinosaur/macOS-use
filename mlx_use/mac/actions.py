# --- START OF FILE mac_use/mac/actions.py ---
import logging
import time

import Cocoa
import Quartz
from ApplicationServices import (
	AXUIElementPerformAction,
	AXUIElementSetAttributeValue,
	kAXPressAction,
	kAXValueAttribute,
	kAXConfirmAction,
	AXUIElementCopyActionNames,
)
from Foundation import NSString

from mlx_use.mac.element import MacElementNode

logger = logging.getLogger(__name__)


def _cg_click_at(x: float, y: float) -> bool:
	"""Perform a real mouse click at screen coordinates using CGEvent (bypasses AXPress limitations on SPAs)."""
	try:
		point = Quartz.CGPointMake(x, y)
		# Mouse down
		event_down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft)
		Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
		time.sleep(0.05)
		# Mouse up
		event_up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft)
		Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
		logger.info(f'✅ CGEvent click at ({x:.0f}, {y:.0f})')
		return True
	except Exception as e:
		logger.error(f'❌ CGEvent click failed at ({x:.0f}, {y:.0f}): {e}')
		return False


def _get_element_center(element: MacElementNode) -> tuple[float, float] | None:
	"""Get the center coordinates of an element from its position and size attributes."""
	pos = element.position
	size = element.size
	if pos and size:
		try:
			px, py = float(pos[0]), float(pos[1])
			sw, sh = float(size[0]), float(size[1])
			if sw > 0 and sh > 0:
				return (px + sw / 2, py + sh / 2)
		except (TypeError, ValueError, IndexError):
			pass
	return None


def perform_action(element: MacElementNode, action: str) -> bool:
	"""Performs a specified accessibility action on an element."""
	try:
		if not element._element:
			logger.error(f'❌ Cannot perform action: Element reference is missing for {element}')
			return False

		# Check if the element supports this action
		available_actions = element.actions
		if action not in available_actions:
			logger.error(f'❌ Action {action} not supported by element {element}. Available actions: {available_actions}')
			return False

		result = AXUIElementPerformAction(element._element, action)
		if result == 0:
			logger.debug(f'✅ Successfully performed {action} on element: {element}')
			return True
		else:
			logger.error(f'❌ Failed to perform {action} on element: {element}, error code: {result}')
			return False
	except Exception as e:
		logger.error(f'❌ Error performing {action} on element: {element}, {e}')
		return False


def click(element: MacElementNode, action: str) -> bool:
	"""Simulates a click on a Mac UI element. Falls back to CGEvent coordinate click if AXPress fails."""
	if not element._element:
		logger.error(f'❌ Cannot click: Element reference is missing for {element}')
		return False

	# Check if element is enabled
	if not element.enabled:
		logger.error(f'❌ Cannot click: Element is disabled: {element}')
		return False

	# Verify element has the press/click action
	actions = ['AXPress', 'AXClick', 'AXOpen', 'AXConfirm', 'AXShowMenu']
	if action not in actions:
		logger.error(f'❌ Cannot click: Element does not support {action} action: {element}')
		return False

	# Try AXPress first
	ax_result = perform_action(element, action)
	if ax_result:
		return True

	# Fallback: CGEvent coordinate click (handles SPAs like YouTube where AXPress doesn't navigate)
	logger.warning(f'⚠️ AXPress failed for {element}, falling back to CGEvent coordinate click')
	center = _get_element_center(element)
	if center:
		return _cg_click_at(center[0], center[1])
	else:
		logger.error(f'❌ CGEvent fallback failed: no position/size for {element}')
		return False


def type_into(element: MacElementNode, text: str, submit: bool = False) -> bool:
	"""Simulates typing text into a Mac UI element with action-based submission"""
	try:
		if not element._element:
			logger.error(f'❌ Cannot type: Element reference is missing for {element}')
			return False

		# Check if element is enabled
		if not element.enabled:
			logger.error(f'❌ Cannot type: Element is disabled: {element}')
			return False

		# Type the text using attribute setting
		ns_string = NSString.stringWithString_(text)
		type_result = AXUIElementSetAttributeValue(element._element, kAXValueAttribute, ns_string)

		if type_result != 0:
			logger.error(f"❌ Failed to type '{text}' into element: {element}, error code: {type_result}")
			return False

		logger.info(f"✅ Successfully typed '{text}' into element: {element}")

		# Handle submission if requested
		if submit:
			available_actions = element.actions
			if 'AXConfirm' in available_actions:
				return perform_action(element, 'AXConfirm')
			elif 'AXPress' in available_actions:
				return perform_action(element, 'AXPress')
			else:
				logger.error(f'❌ No suitable submit action found. Available actions: {available_actions}')
				return False

		return True

	except Exception as e:
		logger.error(f'❌ Error typing into element: {element}, {e}')
		return False


def right_click(element: MacElementNode) -> bool:
	"""Simulates a right-click on a Mac UI element."""
	if not element._element:
		logger.error(f'❌ Cannot right click: Element reference is missing for {element}')
		return False

	# Check if element is enabled
	if not element.enabled:
		logger.error(f'❌ Cannot right click: Element is disabled: {element}')
		return False

	# Check for menu action
	if 'AXShowMenu' in element.actions:
		return perform_action(element, 'AXShowMenu')
	else:
		logger.warning(f'Element does not support AXShowMenu, falling back to regular click for: {element}')
		return click(element)


def scroll(element: MacElementNode, direction: str) -> bool:
	"""
	Scrolls an element in the specified direction.
	direction can be: 'left', 'right', 'up', 'down'
	"""
	direction_map = {
		'left': 'AXScrollLeftByPage',
		'right': 'AXScrollRightByPage',
		'up': 'AXScrollUpByPage',
		'down': 'AXScrollDownByPage',
	}

	if direction not in direction_map:
		logger.error(f'❌ Invalid scroll direction: {direction}')
		return False

	action = direction_map[direction]
	if action in element.actions:
		return perform_action(element, action)
	else:
		logger.error(f'❌ Element does not support scrolling {direction}: {element}')
		return False
