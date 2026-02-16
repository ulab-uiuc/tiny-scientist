import { useEffect } from 'react';
import userActionTracker from '../utils/userActionTracker';

export const useUserActionTracking = (elementRef, targetName, options = {}) => {
  const {
    trackHover = true,
    trackClick = true,
    trackFocus = false,
    trackScroll = false,
    trackDrag = true,
    additionalDetails = {},
    preventDuplicates = false // New option to prevent duplicate tracking
  } = options;

  useEffect(() => {
    const element = elementRef?.current;
    if (!element) return;

    if (preventDuplicates && element.hasAttribute('data-tracking-enabled')) return;
    if (preventDuplicates) element.setAttribute('data-tracking-enabled', 'true');

    const priorTrackingName = element.getAttribute('data-tracking-name');
    if (!priorTrackingName) element.setAttribute('data-tracking-name', targetName);

    if (!trackHover) element.setAttribute('data-uatrack-suppress-hover', 'true');
    if (!trackClick) element.setAttribute('data-uatrack-suppress-click', 'true');
    if (!trackFocus) element.setAttribute('data-uatrack-suppress-focus', 'true');
    if (!trackScroll) element.setAttribute('data-uatrack-suppress-scroll', 'true');
    if (!trackDrag) element.setAttribute('data-uatrack-suppress-drag', 'true');

    userActionTracker.registerElement(element, targetName, additionalDetails);

    return () => {
      userActionTracker.unregisterElement(element);

      if (!priorTrackingName) element.removeAttribute('data-tracking-name');
      element.removeAttribute('data-uatrack-suppress-hover');
      element.removeAttribute('data-uatrack-suppress-click');
      element.removeAttribute('data-uatrack-suppress-focus');
      element.removeAttribute('data-uatrack-suppress-scroll');
      element.removeAttribute('data-uatrack-suppress-drag');

      if (preventDuplicates) element.removeAttribute('data-tracking-enabled');
    };
  }, [elementRef, targetName, trackHover, trackClick, trackFocus, trackScroll, trackDrag, additionalDetails, preventDuplicates]);
};

// Hook for tracking form interactions
export const useFormTracking = (formRef, formName) => {
  useEffect(() => {
    const form = formRef?.current;
    if (!form) return;

    // Suppress tracking entirely for this form; submit will be handled manually.
    form.setAttribute('data-uatrack-suppress', 'true');

    return () => {
      form.removeAttribute('data-uatrack-suppress');
    };
  }, [formRef, formName]);
};

// Hook for tracking navigation
export const useNavigationTracking = () => {
  useEffect(() => {
    // Now handled globally by userActionTracker
    return () => {};
  }, []);
};

// Hook for comprehensive button tracking
export const useButtonTracking = (buttonRef, buttonName, additionalDetails = {}) => {
  useEffect(() => {
    const button = buttonRef?.current;
    if (!button) return;
    const priorTrackingName = button.getAttribute('data-tracking-name');
    if (!priorTrackingName) button.setAttribute('data-tracking-name', buttonName);
    userActionTracker.registerElement(button, buttonName, additionalDetails);

    return () => {
      userActionTracker.unregisterElement(button);
      if (!priorTrackingName) button.removeAttribute('data-tracking-name');
    };
  }, [buttonRef, buttonName, additionalDetails]);
};

// Hook for tracking all buttons in a component automatically
export const useComponentTracking = (componentRef, componentName) => {
  useEffect(() => {
    const component = componentRef?.current;
    if (!component) return;
    userActionTracker.registerElement(component, componentName, { componentName });

    return () => {
      userActionTracker.unregisterElement(component);
    };
  }, [componentRef, componentName]);
};
