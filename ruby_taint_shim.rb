# Ruby 3.2+ removed taint; Liquid 4.x (used by Jekyll 3.9) still calls obj.tainted?
# Define no-ops so Liquid doesn't raise NoMethodError.
if !Object.new.respond_to?(:tainted?, true)
  class Object
    def tainted?
      false
    end

    def taint
      self
    end

    def untaint
      self
    end
  end
end
